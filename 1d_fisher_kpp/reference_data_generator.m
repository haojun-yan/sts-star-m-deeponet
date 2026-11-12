function data = reference_data_generator(cfg, work_dir)
% Generate shared FP32 Fisher--KPP trajectories.
r = cfg.reference_data;
rng(r.data_seed,'twister');
counts = [r.num_of_training_sample,r.num_of_validation_sample,r.num_of_testing_sample];
parameters = zeros(3,3,sum(counts));
parameters(:,1,:) = .25+.60*rand(3,1,sum(counts));
parameters(:,2,:) = -.75+1.5*rand(3,1,sum(counts));
parameters(:,3,:) = .06+.10*rand(3,1,sum(counts));
x = -1+2*(0:r.Nx-1)'/r.Nx;
t = linspace(0,1,r.Nt)';
sensor_coordinates = -1+2*(0:r.nsensor-1)'/r.nsensor;
rawsensors = initial(parameters,sensor_coordinates);
data = struct('x',x,'t',t,'sensor_coordinates',sensor_coordinates, ...
    'grid_shape',[r.Nx,r.Nt],'output_layout','space fastest, then time; columns are cases');
fprintf('[Data] Training: %d | Validation: %d | Test: %d\n',counts);
fprintf('[Data] Grid: Nx %d, Nt %d, sensors %d, reference %d\n',r.Nx,r.Nt,r.nsensor,r.Nref);
fprintf('[Data] Solver: Fourier Strang splitting, diffusion %.9g, reaction %.9g, substeps %d\n', ...
    r.diffusion,r.reaction,r.substeps);
timer = tic;
names = {'train','validation','test'};
offset = 0;
for j = 1:3
    ids = offset+(1:counts(j));
    offset = offset+counts(j);
    file = fullfile(work_dir,['gt_' names{j} '.mat']);
    s = data;
    s.caseids = ids;
    s.rawsensors = rawsensors(:,ids);
    s.initialparameters = parameters(:,:,ids);
    save(file,'-struct','s','-v7.3');
    f = matfile(file,'Writable',true);
    f.u(r.Nx*r.Nt,counts(j)) = single(0);
    for first = 1:r.batch_size:counts(j)
        localids = first:min(first+r.batch_size-1,counts(j));
        fields = reference(parameters(:,:,ids(localids)),r);
        minimum = min(fields,[],'all');
        maximum = max(fields,[],'all');
        assert(all(isfinite(fields),'all') && minimum > -1e-6 && maximum < 1+1e-6, ...
            'FisherKPP:Reference','Reference solution range [%.3e, %.3e] is invalid. Increase Nref or substeps.', ...
            minimum,maximum);
        f.u(:,localids) = fields;
        fprintf('[Data] Progress: %d/%d | Elapsed: %.3f s\n',ids(localids(end)),sum(counts),toc(timer));
    end
    data.(names{j}) = file;
end
data.generation_seconds = toc(timer);
normalization_timer = tic;
data.normalization = struct('mean',single(mean(rawsensors(:,1:r.nfit),2)), ...
    'std',single(std(rawsensors(:,1:r.nfit),0,2)),'epsilon',single(1e-8));
data.normalization_seconds = toc(normalization_timer);
fprintf('[Data] Finished: %.3f s\n',data.generation_seconds);
end

function u = initial(parameters,x)
u = zeros(numel(x),size(parameters,3));
for i = 1:size(parameters,3)
    a = zeros(size(x));
    for j = 1:3
        for shift = -2:2
            a = a+parameters(j,1,i)*exp(-(x-parameters(j,2,i)+2*shift).^2/(2*parameters(j,3,i)^2));
        end
    end
    u(:,i) = 1-exp(-a);
end
end

function fields = reference(parameters,r)
xi = -1+2*(0:r.Nref-1)'/r.Nref;
dt = 1/((r.Nt-1)*r.substeps);
k = pi*[0:r.Nref/2,-r.Nref/2+1:-1]';
dif = exp(-r.diffusion*k.^2*dt);
rh = exp(r.reaction*dt/2);
u = initial(parameters,xi);
fields = zeros(r.Nx*r.Nt,size(parameters,3),'single');
fields(1:r.Nx,:) = single(u(1:r.Nref/r.Nx:end,:));
for frame = 2:r.Nt
    for step = 1:r.substeps
        u = rh*u./(1+(rh-1)*u);
        u = real(ifft(dif.*fft(u,[],1),[],1));
        u = rh*u./(1+(rh-1)*u);
    end
    fields((frame-1)*r.Nx+(1:r.Nx),:) = single(u(1:r.Nref/r.Nx:end,:));
end
end
