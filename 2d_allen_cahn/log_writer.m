function log_writer(run_dir,state)
% Atomically write available results in a compact experiment log.
temporary = [tempname(run_dir) '.txt'];
cleanup = onCleanup(@()remove_temporary(temporary));
fid = fopen(temporary,'w');
assert(fid >= 0,'AC:Log','Cannot open the experiment log.');
close_file = onCleanup(@()fclose(fid));
fprintf(fid,'[Reference Data]\n');
s = state.reference;
entry('status',s.status,'');
mapping = {'data_seed','data_seed';'num_of_training_sample','training_cases'; ...
    'num_of_validation_sample','validation_cases';'num_of_testing_sample','test_cases'};
for j = 1:size(mapping,1)
    if isfield(s,mapping{j,2})
        entry(mapping{j,1},sprintf('%d',s.(mapping{j,2})), '');
    end
end
failure(s);
method('Fixed-Basis SVD DeepONet',state.svd);
method('STS star_M DeepONet',state.sts);
clear close_file;
[ok,message] = movefile(temporary,fullfile(run_dir,'log.txt'),'f');
assert(ok,'AC:Log','%s',message);

    function method(title,s)
        fprintf(fid,'\n[%s]\n',title);
        entry('status',s.status,'');
        fields = {'training_seed','seed','integer','';'rank','rank','integer',''; ...
            'basis_storage','basis_storage_mb','decimal','MB'; ...
            'total_storage','model_storage_mb','decimal','MB'; ...
            'mean_test_relative_l2','test_relative_l2','percent',''; ...
            'total_time','method_seconds','decimal','s'};
        for k = 1:size(fields,1)
            source = fields{k,2};
            if ~isfield(s,source) || ~isscalar(s.(source)) || ~isfinite(s.(source))
                continue;
            end
            value = s.(source);
            switch fields{k,3}
                case 'integer'
                    value = sprintf('%d',value);
                case 'percent'
                    value = sprintf('%.3f %%',100*value);
                otherwise
                    value = sprintf('%.3f',value);
            end
            entry(fields{k,1},value,fields{k,4});
        end
        if strcmp(s.status,'completed')
            entry('cpu_name',state.reference.CPU,'');
            entry('gpu_name',state.reference.GPU,'');
        end
        failure(s);
    end

    function failure(s)
        fields = {'failed_phase','failed_stage';'error_identifier','error_identifier';'error_message','error_message'};
        for k = 1:size(fields,1)
            if isfield(s,fields{k,1})
                entry(fields{k,2},s.(fields{k,1}),'');
            end
        end
        if isfield(s,'elapsed_seconds')
            entry('elapsed_time',sprintf('%.3f',s.elapsed_seconds),'s');
        end
    end

    function entry(key,value,unit)
        value = regexprep(char(string(value)),'[\r\n]+',' ');
        line = sprintf('%s: %s',key,value);
        if isempty(unit)
            fprintf(fid,'%s\n',line);
        else
            fprintf(fid,'%-60s# unit: %s\n',line,unit);
        end
    end
end

function remove_temporary(file)
if isfile(file)
    delete(file);
end
end
