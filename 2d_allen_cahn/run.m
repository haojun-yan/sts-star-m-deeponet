function varargout = run(cfg)
% Run one shared-data, single-seed Allen--Cahn experiment.
assert(nargout <= 1,'AC:Outputs','At most one result output is supported.');
wall = tic;
fprintf('=== 2D Allen--Cahn ===\n');
fprintf('[Start] Local time: %s\n',char(datetime('now','Format','yyyy-MM-dd HH:mm:ss')));
if nargin == 0
    cfg = config();
    fprintf('[Config] Loaded config.m\n');
else
    fprintf('[Config] Using supplied configuration\n');
end
previous_rng = rng;
previous_threads = maxNumCompThreads;
previous_path = path;
previous_dir = pwd;
run_dir = '';
created_run_dir = false;
resources = containers.Map('KeyType','char','ValueType','any');
resources('pool') = [];
resources('work_dir') = '';
resources('gpu_rng') = [];
resources('deterministic') = [];
resources('compatibility') = [];
resources('threads') = previous_threads;
resources('rng') = previous_rng;
resources('path') = previous_path;
resources('directory') = previous_dir;
session_cleanup = onCleanup(@()cleanup_resources(resources));
state = struct('reference',struct('status','not_run'),'svd',struct('status','not_run'), ...
    'sts',struct('status','not_run'));
if ~cfg.common.run_svd_deeponet
    state.svd.status = 'disabled';
end
if ~cfg.common.run_sts_deeponet
    state.sts.status = 'disabled';
end
phase = 'preflight';
section = 'reference';
try
    fprintf('[Check] MATLAB release: %s\n',version('-release'));
    validate_config(cfg);
    r = cfg.reference_data;
    train = cfg.common.run_svd_deeponet || cfg.common.run_sts_deeponet;
    methods = {'Fixed-Basis SVD DeepONet','STS star_M DeepONet'};
    methods = methods([cfg.common.run_svd_deeponet,cfg.common.run_sts_deeponet]);
    if isempty(methods)
        methods = {'reference data only'};
    end
    fprintf('[Config] Methods: %s\n',strjoin(methods,', '));
    fprintf('[Config] Data seed: %d | Training seeds: SVD=%d, STS=%d\n', ...
        r.data_seed,cfg.svd.training_seed,cfg.sts.training_seed);
    fprintf('[Check] Configuration and grid constraints: passed\n');
    assert(license('test','Distrib_Computing_Toolbox') && ~isempty(ver('parallel')), ...
        'AC:Toolbox','Parallel Computing Toolbox is required.');
    fprintf('[Check] Parallel Computing Toolbox: available\n');
    if train
        assert(license('test','Neural_Network_Toolbox') && ~isempty(ver('nnet')), ...
            'AC:Toolbox','Deep Learning Toolbox is required for training.');
        assert(~isempty(which('lbfgsState')) && ~isempty(which('lbfgsupdate')), ...
            'AC:Release','This MATLAB release must provide lbfgsState and lbfgsupdate.');
        fprintf('[Check] Deep Learning Toolbox: available\n');
    else
        fprintf('[Check] Deep Learning Toolbox and GPU: skipped (data only)\n');
    end
    [cores,cpu_name] = physical_cores();
    if ispc
        operating_system = 'Windows';
    elseif ismac
        operating_system = 'macOS';
    else
        operating_system = 'Linux';
    end
    fprintf('[Check] Operating system: %s\n[Check] CPU: %s\n[Check] Physical CPU cores: %d\n', ...
        operating_system,cpu_name,cores);
    pool = gcp('nocreate');
    if ~isempty(pool)
        compatible = isa(pool,'parallel.ThreadPool') || ...
            (isa(pool,'parallel.ProcessPool') && isa(pool.Cluster,'parallel.cluster.Local'));
        assert(compatible,'AC:Pool','Existing pool must be a local thread or process pool.');
        assert(pool.NumWorkers == cores,'AC:Pool', ...
            'Existing pool has %d workers; %d physical-core workers are required.',pool.NumWorkers,cores);
        fprintf('[Check] Parallel pool: reusing %d workers\n',cores);
    end
    gpu_name = 'CPU';
    if train && cfg.common.use_gpu
        phase = 'gpu_initialization';
        previous_compat = parallel.gpu.enableCUDAForwardCompatibility;
        try
            g = gpuDevice;
        catch err
            if contains(err.identifier,'DeviceTooNew')
                parallel.gpu.enableCUDAForwardCompatibility(true);
                compat_changed = ~previous_compat;
                if compat_changed
                    resources('compatibility') = previous_compat;
                end
                fprintf('[Check] CUDA forward compatibility: enabled after DeviceTooNew\n');
                g = gpuDevice;
            else
                rethrow(err);
            end
        end
        gpu_name = g.Name;
        gpu_rng = gpurng;
        resources('gpu_rng') = gpu_rng;
        if ~isempty(which('deep.gpu.deterministicAlgorithms'))
            previous_deterministic = deep.gpu.deterministicAlgorithms;
            resources('deterministic') = previous_deterministic;
            deep.gpu.deterministicAlgorithms(true);
            fprintf('[Check] GPU deterministic algorithms: enabled\n');
        end
        fprintf('[Check] GPU: %s; CUDA forward compatibility %s\n',g.Name,enabled(parallel.gpu.enableCUDAForwardCompatibility));
    elseif train
        fprintf('[Check] GPU: disabled; CPU training enabled\n');
    end
    phase = 'preflight';
    if isempty(pool)
        owned_pool = parpool('Threads',cores);
        resources('pool') = owned_pool;
        fprintf('[Check] Parallel pool: created %d workers\n',cores);
    end
    root = fileparts(mfilename('fullpath'));
    run_dir = fullfile(root,'RUNs',char(datetime('now','Format','yyyyMMdd_HHmm')));
    assert(~isfolder(run_dir) && ~isfile(run_dir),'AC:DuplicateRun', ...
        'Output directory already exists: %s',run_dir);
    [ok,message] = mkdir(run_dir);
    assert(ok,'AC:Output','%s',message);
    created_run_dir = true;
    [~,run_name] = fileparts(run_dir);
    fprintf('[Output] Created RUNs/%s/\n',run_name);
    work_dir = tempname;
    mkdir(work_dir);
    resources('work_dir') = work_dir;
    fprintf('[Data] Temporary directory: %s\n',work_dir);
    admit_resources(cfg,work_dir,run_dir,train);
    flags = fieldnames(cfg.common);
    for j = 1:numel(flags)
        if startsWith(flags{j},'save_')
            fprintf('[Config] %s: %s\n',flags{j},enabled(cfg.common.(flags{j})));
        end
    end
    state.reference = struct('status','running','data_seed',r.data_seed,'Nx',r.Nx,'Ny',r.Ny,'Nt',r.Nt, ...
        'Nref',r.Nref,'sensors',r.nsensor,'training_cases',r.num_of_training_sample, ...
        'validation_cases',r.num_of_validation_sample,'test_cases',r.num_of_testing_sample, ...
        'fit_cases',r.nfit,'physical_cores',cores,'CPU',cpu_name,'GPU',gpu_name);
    update_log();
    phase = 'reference_generation';
    fprintf('[Data] Generating reference solutions\n');
    data = reference_data_generator(cfg,work_dir);
    fprintf('[Data] Reference solutions completed\n');
    state.reference.status = 'completed';
    state.reference.generation_seconds = data.generation_seconds;
    for name = {'train','validation','test'}
        target = ['gt_' name{1} '.mat'];
        if cfg.common.(['save_gt_' name{1}])
            atomic_copy(data.(name{1}),fullfile(run_dir,target));
            fprintf('[Save] %s: saved\n',target);
        else
            fprintf('[Save] %s: disabled\n',target);
        end
    end
    update_log();
    for method = {'svd','sts'}
        section = method{1};
        if cfg.common.(['run_' section '_deeponet'])
            phase = [section '_training_export_test'];
            state.(section) = struct('status','running','seed',cfg.(section).training_seed);
            update_log();
            execute = str2func([section '_benchmark']);
            state.(section) = execute('train',cfg,data,run_dir);
            update_log();
        else
            fprintf('[%s] Disabled\n',upper(section));
        end
    end
    phase = 'output_check';
    check_outputs(cfg,run_dir);
    fprintf('[Check] Requested output files: passed\n');
    cleanup_session();
    fprintf('[Cleanup] Temporary working files removed\n');
    result = state;
    result.run_dir = run_dir;
    result.total_wall_seconds = toc(wall);
    fprintf('[Done] Total wall time: %.3f s\n[Done] Results: %s\n',result.total_wall_seconds,run_dir);
    if nargout == 1
        varargout{1} = result;
    end
catch err
    state.(section).status = 'failed';
    details = regexp(err.identifier,'^AC:(SVD|STS)_(.+)$','tokens','once');
    if ~isempty(details)
        phase = details{2};
    end
    state.(section).failed_phase = phase;
    state.(section).error_identifier = err.identifier;
    if ~isempty(details) && ~isempty(err.cause)
        state.(section).error_identifier = err.cause{1}.identifier;
    end
    state.(section).error_message = err.message;
    state.(section).elapsed_seconds = toc(wall);
    if created_run_dir && isfolder(run_dir)
        try
            update_log();
        catch log_error
            fprintf('Log update failed: %s\n',log_error.message);
        end
    end
    cleanup_session();
    fprintf('[Error] Stopped in %s: %s\n',phase,err.message);
    if strcmp(phase,'gpu_initialization')
        fprintf('[Action] Set common.use_gpu = false in config.m to run on CPU.\n');
    end
    fprintf('[Cleanup] Temporary working files removed\n[Done] Total wall time: %.3f s\n',toc(wall));
    if ~isempty(run_dir)
        fprintf('[Done] Results: %s\n',run_dir);
    end
    rethrow(err);
end

    function update_log()
        if cfg.common.save_log
            log_writer(run_dir,state);
            fprintf('[Log] log.txt updated\n');
        end
    end

    function cleanup_session()
        cleanup_resources(resources);
    end
end

function cleanup_resources(resources)
pool = resources('pool');
if ~isempty(pool) && isvalid(pool)
    delete(pool);
end
resources('pool') = [];
folder = resources('work_dir');
if ~isempty(folder) && isfolder(folder)
    rmdir(folder,'s');
end
resources('work_dir') = '';
value = resources('gpu_rng');
if ~isempty(value)
    gpurng(value);
end
resources('gpu_rng') = [];
value = resources('deterministic');
if ~isempty(value)
    deep.gpu.deterministicAlgorithms(value);
end
resources('deterministic') = [];
value = resources('compatibility');
if ~isempty(value)
    parallel.gpu.enableCUDAForwardCompatibility(value);
end
resources('compatibility') = [];
maxNumCompThreads(resources('threads'));
rng(resources('rng'));
if ~strcmp(pwd,resources('directory'))
    cd(resources('directory'));
end
if ~strcmp(path,resources('path'))
    path(resources('path'));
end
end

function validate_config(cfg)
c = cfg.common;
for name = fieldnames(c)'
    validateattributes(c.(name{1}),{'logical'},{'scalar'},mfilename,['common.' name{1}]);
end
r = cfg.reference_data;
for name = {'Nx','Ny','Nt','Nref','nsensor','num_of_training_sample','num_of_validation_sample','num_of_testing_sample','nfit','batch_size'}
    validateattributes(r.(name{1}),{'numeric'},{'real','finite','scalar','integer','positive'});
end
validateattributes(r.data_seed,{'numeric'},{'real','finite','scalar','integer','nonnegative','<=',2^32-1});
validateattributes(r.dt,{'numeric'},{'real','finite','scalar','positive','<=',1});
validateattributes(r.kappa,{'numeric'},{'real','finite','scalar','positive'});
assert(r.Nx >= 2 && r.Ny == r.Nx && r.Nt >= 2 && mod(r.Nref,2) == 0 && mod(r.Nref,r.Nx) == 0, ...
    'AC:Grid','Reference grid must be even and nest the output grid.');
assert(r.Nref >= 16 && r.nsensor == 256,'AC:Grid','Require at least 16 reference points and fixed 16-by-16 sensors.');
assert(r.nfit <= r.num_of_training_sample,'AC:Fit','nfit exceeds training count.');
nsteps = round(1/r.dt);
assert(abs(nsteps*r.dt-1) < 1e-10 && mod(nsteps,r.Nt-1) == 0, ...
    'AC:TimeGrid','Output times and final time must coincide with reference steps.');
for method = {'svd','sts'}
    q = cfg.(method{1});
    validateattributes(q.training_seed,{'numeric'},{'real','finite','scalar','integer','nonnegative','<=',2^32-1});
    for name = {'width','total_epoch','lr_decay_step','lbfgs_updates','lbfgs_gradient_calls','vp_updates','vp_gradient_calls','lbfgs_history','lbfgs_linesearch'}
        validateattributes(q.(name{1}),{'numeric'},{'real','finite','scalar','integer','positive'});
    end
    for name = {'initial_lr','epsilon','lbfgs_tolerance'}
        validateattributes(q.(name{1}),{'numeric'},{'real','finite','scalar','positive'});
    end
    for name = {'energy','lr_decay_ratio'}
        validateattributes(q.(name{1}),{'numeric'},{'real','finite','scalar','positive','<=',1});
    end
    for name = {'beta1','beta2'}
        validateattributes(q.(name{1}),{'numeric'},{'real','finite','scalar','>=',0,'<',1});
    end
    assert(ismember(q.alpha,[0 1]),'AC:Alpha','alpha must be zero or one.');
end
end

function [cores,name] = physical_cores()
if ispc
    name = winqueryreg('HKEY_LOCAL_MACHINE','HARDWARE\DESCRIPTION\System\CentralProcessor\0','ProcessorNameString');
    name = strtrim(name);
    lines = {
        '$ErrorActionPreference = "Stop"'
        '$ProgressPreference = "SilentlyContinue"'
        'Add-Type -TypeDefinition @"'
        'using System;'
        'using System.Runtime.InteropServices;'
        'public static class AcCpuTopology {'
        ' [DllImport("kernel32.dll", SetLastError=true)]'
        ' static extern bool GetLogicalProcessorInformationEx(int relationship, IntPtr buffer, ref uint length);'
        ' public static int PhysicalCores() {'
        '  uint length=0; GetLogicalProcessorInformationEx(0,IntPtr.Zero,ref length);'
        '  if(length==0) throw new Exception("Cannot size CPU topology.");'
        '  IntPtr p=Marshal.AllocHGlobal((int)length);'
        '  try {'
        '   if(!GetLogicalProcessorInformationEx(0,p,ref length)) throw new Exception("Cannot read CPU topology.");'
        '   int count=0; long offset=0;'
        '   while(offset<length) {'
        '    IntPtr item=new IntPtr(p.ToInt64()+offset);'
        '    int relation=Marshal.ReadInt32(item,0), size=Marshal.ReadInt32(item,4);'
        '    if(size<8 || offset+size>length) throw new Exception("Invalid CPU topology record.");'
        '    if(relation==0) count++; offset+=size;'
        '   } return count;'
        '  } finally { Marshal.FreeHGlobal(p); }'
        ' }'
        '}'
        '"@'
        '[AcCpuTopology]::PhysicalCores()'};
    script = strjoin(lines,newline);
    encoded = matlab.net.base64encode(unicode2native(script,'UTF-16LE'));
    [status,output] = system(sprintf('powershell.exe -NoLogo -NoProfile -NonInteractive -EncodedCommand %s',encoded));
    assert(status == 0,'AC:CPU','CPU topology query failed: %s',output);
    cores = str2double(strtrim(output));
elseif ismac
    [status,output] = system('sysctl -n hw.physicalcpu');
    assert(status == 0,'AC:CPU','Physical CPU query failed.');
    cores = str2double(strtrim(output));
    [cpu_status,name] = system('/usr/sbin/sysctl -n machdep.cpu.brand_string 2>/dev/null');
    if cpu_status ~= 0 || isempty(strtrim(name))
        [~,hardware] = system('LC_ALL=C /usr/sbin/system_profiler SPHardwareDataType');
        chip = regexp(hardware,'Chip:\s*([^\r\n]+)','tokens','once');
        if isempty(chip)
            [~,hardware_model] = system('/usr/sbin/sysctl -n hw.model');
            name = ['Unavailable (hardware model: ' strtrim(hardware_model) ')'];
        else
            name = chip{1};
        end
    end
    name = strtrim(name);
else
    [status,output] = system('lscpu -p=CORE,SOCKET');
    assert(status == 0,'AC:CPU','Physical CPU query failed.');
    lines = splitlines(string(output));
    lines = lines(~startsWith(lines,'#') & strlength(lines)>0);
    cores = numel(unique(lines));
    name = 'Linux CPU';
end
assert(isfinite(cores) && cores >= 1 && fix(cores) == cores,'AC:CPU','Invalid physical CPU count.');
end

function text = enabled(value)
if value
    text = 'enabled';
else
    text = 'disabled';
end
end

function atomic_copy(source,target)
temporary = [tempname(fileparts(target)) '.mat'];
cleanup = onCleanup(@()delete_if_present(temporary));
[ok,message] = copyfile(source,temporary);
assert(ok,'AC:Save','%s',message);
[ok,message] = movefile(temporary,target,'f');
assert(ok,'AC:Save','%s',message);
end

function delete_if_present(file)
if isfile(file)
    delete(file);
end
end

function check_outputs(cfg,folder)
expected = {};
for name = {'train','validation','test'}
    if cfg.common.(['save_gt_' name{1}])
        expected{end+1} = ['gt_' name{1} '.mat']; %#ok<AGROW>
    end
end
for method = {'svd','sts'}
    if cfg.common.(['run_' method{1} '_deeponet'])
        for name = {'model','prediction','relative_l2_vs_time'}
            if cfg.common.(['save_' method{1} '_' name{1}])
                expected{end+1} = [method{1} '_' name{1} '.mat']; %#ok<AGROW>
            end
        end
    end
end
if cfg.common.save_log
    expected{end+1} = 'log.txt';
end
files = dir(folder);
actual = {files(~[files.isdir]).name};
assert(isequal(sort(actual),sort(expected)),'AC:Artifacts','Output files differ from enabled saves.');
end





function admit_resources(cfg,work_dir,run_dir,train)
r = cfg.reference_data;
rows = r.Nx*r.Ny*r.Nt;
counts = [r.num_of_training_sample,r.num_of_validation_sample,r.num_of_testing_sample];
reference_bytes = 4*rows*sum(counts);
scratch_bytes = reference_bytes;
if train
    scratch_bytes = scratch_bytes+8*rows*r.nfit;
end
saved_bytes = 0;
names = {'train','validation','test'};
for j = 1:3
    if cfg.common.(['save_gt_' names{j}])
        saved_bytes = saved_bytes+4*rows*counts(j);
    end
end
required_disk = ceil(1.15*(scratch_bytes+saved_bytes)+256*1024^2);
disk = java.io.File(work_dir);
output_disk = java.io.File(run_dir);
assert(double(disk.getUsableSpace()) >= required_disk && double(output_disk.getUsableSpace()) >= required_disk, ...
    'AC:Disk','Insufficient free disk space: require %.2f GiB.',required_disk/1024^3);
if train && ispc
    [~,system_memory] = memory;
    required_ram = 1024^3+24*rows*r.nfit;
    assert(system_memory.PhysicalMemory.Available >= required_ram,'AC:Memory', ...
        'Insufficient available RAM for basis construction: require %.2f GiB.',required_ram/1024^3);
end
fprintf('[Check] Resource admission: passed; disk reserve %.2f GiB\n',required_disk/1024^3);
end
