function result = sts_benchmark(action, cfg, data, run_dir)
% Train a fixed-basis model or predict from raw sensor columns.
if strcmpi(action,'predict')
    result = predict_model(cfg,data);
    return;
end
assert(strcmpi(action,'train'),'AC:Action','Use train or predict.');
method = 'sts';
q = cfg.(method);
r = cfg.reference_data;
gpu = cfg.common.use_gpu;
phase = 'basis';
try
timer = tic;
fprintf('\n[STS] Building slice bases and the temporal basis\n');
[basis,trainc,valc,~] = prepare_basis(data,r,q,method);
prep_seconds = data.normalization_seconds+toc(timer);
fprintf('[STS] Basis completed | Temporal rank: %d | Elapsed: %.3f s\n',size(basis,2),prep_seconds);
z = load(data.train,'rawsensors');
X = double(normalize_input(z.rawsensors,data.normalization));
z = load(data.validation,'rawsensors');
Vraw = z.rawsensors;
V = double(normalize_input(Vraw,data.normalization));
mu = double(trainc.mu);
cs = double(trainc.scale);
ct = trainc.target;
R = struct('R',trainc.R,'ns',trainc.ns);
phase = 'initialization';
fprintf('[STS] Initializing the branch network | Seed: %d\n',q.training_seed);
if gpu
    gpurng(q.training_seed);
end
b = networks(size(ct,1),q.training_seed,r.nsensor,mu,q.width);
if q.alpha == 1
    b = initialize_affine(b,X,ct,R,mu,cs,0);
end
b = netmove(b,@(v)devicearray(single(v),gpu));
A = dlarray(devicearray(single(X),gpu),'CB');
VA = dlarray(devicearray(single(V),gpu),'CB');
gc = move_fields(trainc,gpu);
gv = move_fields(valc,gpu);
ab = []; vb = [];
fast_loss = dlaccelerate(@coefficient_loss);
phase = 'adam';
for epoch = 1:q.total_epoch
    [loss,gradient] = dlfeval(fast_loss,b,A,gc);
    assert(isfinite(number(loss)),'AC:Loss','Nonfinite Adam loss.');
    lr = q.initial_lr*q.lr_decay_ratio^((epoch-1)/q.lr_decay_step);
    [b,ab,vb] = adamupdate(b,gradient,ab,vb,epoch,lr,q.beta1,q.beta2,q.epsilon);
    if epoch == 1 || mod(epoch,1000) == 0 || epoch == q.total_epoch
        fprintf('[STS][Adam] Epoch %d/%d | LR: %s | Training loss: %s | Validation loss: %s\n', ...
            epoch,q.total_epoch,sci(lr),sci(number(coefficient_value(b,A,gc,[]))),sci(number(coefficient_value(b,VA,gv,[]))));
    end
end
fprintf('[STS][Adam] Finished | Reason: epoch_budget\n');
phase = 'refit';
clear fast_loss ab vb gradient;
b = refit(b,X,ct,R,mu,cs,gpu);
fprintf('[STS][Readout] Refitting the output head\n');
phase = 'lbfgs';
[joint_updates,joint_calls,joint_reason] = optimize(false);
if q.alpha == 1
    b = refit(b,X,ct,R,mu,cs,gpu);
end
phase = 'vp';
vpdata = vp_prepare(X,ct,trainc.floor,trainc.normalizer,0);
template = netmove(b,@gather);
[vp_updates,vp_calls,vp_reason] = optimize(true);
syncgpu(gpu);
method_seconds = data.normalization_seconds+toc(timer);
phase = 'export';
model = struct('branch',numeric_net(b),'basis',single(basis),'method',method, ...
    'mu',single(mu),'coefficient_scale',single(cs),'normalization',data.normalization, ...
    'x',data.x,'y',data.y,'t',data.t,'sensor_coordinates',data.sensor_coordinates, ...
    'grid_shape',data.grid_shape,'output_layout',data.output_layout, ...
    'seed',q.training_seed,'selected_stage','vp_last','selection_policy','terminal_iterate', ...
    'numerical_policy',numerical_policy(),'locked',false);
model.final_adam_epoch = q.total_epoch;
model.test_access_after_lock = true;
model.stage_updates = struct('joint',joint_updates,'vp',vp_updates);
model.stage_gradient_calls = struct('joint',joint_calls,'vp',vp_calls);
model.stage_stop_reasons = struct('joint',joint_reason,'vp',vp_reason);
model.training_loss = number(coefficient_value(b,A,gc,[]));
model.validation_loss = number(coefficient_value(b,VA,gv,[]));
model.training_numerics = readout_diagnostics(b,X,ct,R,mu,cs,trainc);
nativeC = double(cs).*native_export_forward(b,VA)+double(mu);
native = decode(model,nativeC);
deployed = predict_model(model,Vraw);
model.export_relative_difference = norm(native(:)-deployed(:))/max(norm(native(:)),eps);
assert(model.export_relative_difference < 2e-5,'AC:Export','Native/numeric FP32 export disagreement.');
model.locked = true;
fprintf('[STS] Terminal model locked | Export consistency: passed\n');
save_option(['save_' method '_model'],[method '_model.mat'],struct('model',model));
clear A VA gc gv native deployed;
phase = 'test';
fprintf('[STS][Test] Evaluating %d test samples\n',r.num_of_testing_sample);
z = load(data.test);
prediction = predict_model(model,z.rawsensors);
truth = double(z.u);
E = prediction-truth;
case_l2 = sqrt(sum(E.^2,1)./max(sum(truth.^2,1),eps));
ratios = sqrt(sum(reshape(E,r.Nx*r.Ny,r.Nt,[]).^2,1)./max(sum(reshape(truth,r.Nx*r.Ny,r.Nt,[]).^2,1),eps));
mean_relative_l2_vs_time = reshape(mean(ratios,3),[],1);
test_l2 = mean(case_l2);
fprintf('[STS][Test] Mean trajectory relative L2: %.3f %%\n',100*test_l2);
s = struct('prediction',single(prediction),'caseids',z.caseids,'rawsensors',z.rawsensors, ...
    'sensor_coordinates',data.sensor_coordinates,'x',data.x,'y',data.y,'t',data.t, ...
    'grid_shape',data.grid_shape,'output_layout',data.output_layout);
save_option(['save_' method '_prediction'],[method '_prediction.mat'],s);
save_option(['save_' method '_relative_l2_vs_time'],[method '_relative_l2_vs_time.mat'], ...
    struct('time',data.t,'mean_relative_l2',mean_relative_l2_vs_time));
np = numeric_count(model.branch);
extra = numel(model.mu)+numel(model.coefficient_scale)+2*r.nsensor+1;
result = struct('status','completed','seed',q.training_seed,'rank',size(basis,2), ...
    'basis_storage_mb',numel(basis)*4/1e6,'model_storage_mb',(numel(basis)+np+extra)*4/1e6, ...
    'num_parameters',np,'basis_seconds',prep_seconds,'method_seconds',method_seconds, ...
    'test_relative_l2',test_l2,'training_loss',model.training_loss,'validation_loss',model.validation_loss, ...
    'joint_updates',joint_updates,'joint_gradient_calls',joint_calls,'joint_reason',joint_reason, ...
    'vp_updates',vp_updates,'vp_gradient_calls',vp_calls,'vp_reason',vp_reason, ...
    'export_relative_difference',model.export_relative_difference);
fprintf('[STS] Method time: %.3f s\n',method_seconds);
catch err
    failure = MException(['AC:STS_' phase],'%s',err.message);
    failure = addCause(failure,err);
    throw(failure);
end

    function save_option(flag,name,s)
        if cfg.common.(flag)
            atomic(fullfile(run_dir,name),s);
            fprintf('[Save] %s: saved\n',name);
        else
            fprintf('[Save] %s: disabled\n',name);
        end
    end

    function [accepted,count,reason] = optimize(isvp)
        if isvp
            label = 'VP';
            max_updates = q.vp_updates;
            max_calls = q.vp_gradient_calls;
            featureNet = netmove(feature_net(b),@(v)devicearray(double(v),gpu));
            AX = dlarray(devicearray(double(X),gpu),'CB');
            dd = move_fields(vpdata,gpu,false);
            param = pack(featureNet,[]);
        else
            label = 'L-BFGS';
            max_updates = q.lbfgs_updates;
            max_calls = q.lbfgs_gradient_calls;
            featureNet = [];
            param = pack(b,[]);
        end
        solver = lbfgsState(HistorySize=q.lbfgs_history,InitialInverseHessianFactor=1);
        accepted = 0; count = 0; last_report = -1; reason = 'update_budget';
        for step = 1:max_updates
            if count >= max_calls
                reason = 'evaluation_budget'; break;
            end
            previous = param;
            try
                [candidate,next] = lbfgsupdate(param,@counted,solver,LineSearchMethod='strong-wolfe', ...
                    MaxNumLineSearchIterations=q.lbfgs_linesearch);
            catch err
                if strcmp(err.identifier,'AC:Budget')
                    param = previous;
                    reason = 'evaluation_budget'; break;
                end
                rethrow(err);
            end
            if string(next.LineSearchStatus) == "failed"
                param = previous;
                reason = 'line_search_failed'; break;
            end
            param = candidate;
            solver = next;
            accepted = step;
            if accepted == 1 || mod(accepted,50) == 0
                report();
            end
            if number(solver.GradientsNorm) < q.lbfgs_tolerance
                reason = 'gradient_tolerance'; break;
            elseif number(solver.StepNorm) < q.lbfgs_tolerance
                reason = 'step_tolerance'; break;
            end
        end
        if last_report ~= accepted
            report();
        end
        assemble_current();
        fprintf('[STS][%s] Finished | Reason: %s\n',label,reason);

        function [f,g] = counted(v)
            if count >= max_calls
                error('AC:Budget','Loss-gradient evaluation budget exhausted.');
            end
            count = count+1;
            if isvp
                [ff,~] = unpack(v,featureNet,[]);
                [f,gg] = dlfeval(@vp_loss,ff,AX,dd);
            else
                [bb,~] = unpack(v,b,[]);
                [f,gg] = dlfeval(@coefficient_loss,bb,A,gc);
            end
            g = pack(gg,[]);
        end

        function report()
            if isvp
                [ff,~] = unpack(param,featureNet,[]);
                H = extractdata(stripdims(forward(ff,AX)));
                retained = projected_value(H,dd);

            else
                [bb,~] = unpack(param,b,[]);
                retained = number(coefficient_value(bb,A,gc,[]));
            end
            fprintf('[STS][%s] Accepted updates: %d/%d | Loss: %s\n', ...
                label,accepted,max_updates,sci(number(retained)));
            last_report = accepted;
        end

        function assemble_current()
            if isvp
                [featureNet,~] = unpack(param,featureNet,[]);
                b = assemble(template,featureNet,X,ct,R,mu,cs,gpu);
            else
                [b,~] = unpack(param,b,[]);
            end
        end
    end
end

function [basis,trainc,valc,variance] = prepare_basis(data,r,q,method)
f = matfile(data.train);
v = matfile(data.validation);
total = 0; squares = 0;
for it = 1:r.Nt
    rows = (it-1)*(r.Nx*r.Ny)+(1:(r.Nx*r.Ny));
    A = double(f.u(rows,1:r.nfit));
    total = total+sum(A,'all');
    squares = squares+sum(A.^2,'all');
end
count = (r.Nx*r.Ny)*r.Nt*r.nfit;
variance = (squares-total^2/count)/(count-1);
assert(isfinite(variance) && variance > 0,'AC:Variance','Training variance must be positive.');
if strcmp(method,'svd')
    G = zeros(r.nfit);
    for it = 1:r.Nt
        A = double(f.u((it-1)*(r.Nx*r.Ny)+(1:(r.Nx*r.Ny)),1:r.nfit));
        G = G+A'*A;
    end
    [W,s] = spectrum(G);
    rank = energy_rank(s,q.energy);
    basis = zeros((r.Nx*r.Ny)*r.Nt,rank,'single');
    for it = 1:r.Nt
        rows = (it-1)*(r.Nx*r.Ny)+(1:(r.Nx*r.Ny));
        basis(rows,:) = single(double(f.u(rows,1:r.nfit))*(W(:,1:rank)./s(1:rank)'));
    end
    temporal = false;
else
    % Keep exact filtered slices on disk to bound working memory.
    scratch = [tempname(fileparts(data.train)) '.mat'];
    scratch_cleanup = onCleanup(@()delete_if_present(scratch));
    filtered = matfile(scratch,'Writable',true);
    filtered.F(r.Nx*r.Ny,r.nfit,r.Nt) = 0;
    for it = 1:r.Nt
        rows = (it-1)*r.Nx*r.Ny+(1:r.Nx*r.Ny);
        filtered.F(:,:,it) = filter_slice(double(f.u(rows,1:r.nfit)),q.energy);
    end
    G = zeros(r.Nt);
    for first = 1:64:r.Nx*r.Ny
        rows = first:min(first+63,r.Nx*r.Ny);
        Ft = reshape(permute(filtered.F(rows,:,:),[3 1 2]),r.Nt,[]);
        G = G+Ft*Ft';
    end
    clear filtered Ft scratch_cleanup;
    [B,s] = spectrum(G);
    rank = energy_rank(s,q.energy);
    basis = single(B(:,1:rank));
    temporal = true;
end
[Q,R] = qr(double(basis),0);
p = struct('nspace',(r.Nx*r.Ny),'nt',r.Nt);
trainc = project_cache(f,Q,R,p,variance,temporal);
valc = project_cache(v,Q,R,p,variance,temporal);
C = solve_geometry(trainc.target,struct('R',R,'ns',trainc.ns));
[mu,cs] = coefficient_statistics(C(:,1:r.nfit),q.alpha);
trainc.mu = mu; trainc.scale = cs;
valc.mu = mu; valc.scale = cs;
end

function b = networks(p,seed,ns,~,width)
rng(seed,'twister');
body = [
	featureInputLayer(ns, Normalization = 'none', Name = 'input');
	functionLayer(@sensor_image, Name = 'sensor_image', Formattable = true, Acceleratable = true);
	convolution2dLayer(2, 2, Padding = 'same', Name = 'conv_1');
	tanhLayer(Name = 'tanh_1');
	averagePooling2dLayer(2, Stride = 2, Name = 'pool_1');
	convolution2dLayer(2, 4, Padding = 'same', Name = 'conv_2');
	tanhLayer(Name = 'tanh_2');
	averagePooling2dLayer(2, Stride = 2, Name = 'pool_2');
	flattenLayer(Name = 'flatten');
	fullyConnectedLayer(width, Name = 'cnn_dense');
	tanhLayer(Name = 'features')
];
b = dlnetwork([body;fullyConnectedLayer(p,Name='head')]);
b = addLayers(b,fullyConnectedLayer(p,Name='linear_skip'));
b = addLayers(b,additionLayer(2,Name='combined'));
b = connectLayers(b,'input','linear_skip');
b = connectLayers(b,'head','combined/in1');
b = connectLayers(b,'linear_skip','combined/in2');
b = initialize(b);
rng(seed,'twister');
for layer = {'conv_1','conv_2','cnn_dense','head','linear_skip'}
    for j = 1:height(b.Learnables)
        if ~strcmp(b.Learnables.Layer{j},layer{1})
            continue;
        end
        v = extractdata(b.Learnables.Value{j});
        if strcmp(b.Learnables.Parameter{j},'Weights') && ~strcmp(layer{1},'linear_skip')
            if startsWith(layer{1},'conv_')
                fan = size(v,1)*size(v,2)*(size(v,3)+size(v,4));
            else
                fan = size(v,1)+size(v,2);
            end
            v = single((2*rand(size(v))-1)*sqrt(6/fan));
        else
            v = zeros(size(v),'single');
        end
        if strcmp(layer{1},'head')
            v = zeros(size(v),'single');
        end
        b.Learnables.Value{j} = dlarray(v);
    end
end
end

function Y = sensor_image(X)
Y = dlarray(reshape(stripdims(X),16,16,1,[]),'SSCB');
end

function prediction = predict_model(model,sensors)
assert(isnumeric(sensors) && isreal(sensors) && ismatrix(sensors) && size(sensors,1) == size(model.sensor_coordinates,1), ...
    'AC:Sensors','Sensors must have one row per fixed sensor and one column per case.');
assert(all(isfinite(sensors),'all'),'AC:Sensors','Sensors must be finite.');
C = double(model.coefficient_scale).*numeric_forward(model.branch,normalize_input(sensors,model.normalization))+double(model.mu);
prediction = decode(model,C);
end

function Y = decode(model,C)
if strcmp(model.method,'svd')
    Y = double(model.basis)*C;
else
    Y = decode_time(double(model.basis),C,numel(model.x)*numel(model.y),numel(model.t));
end
end

function Y = numeric_forward(s,X)
n = size(X,2);
A = reshape(X,16,16,1,n);
A = pool2(tanh(conv_same(A,s.conv_1)));
A = pool2(tanh(conv_same(A,s.conv_2)));
H = tanh(s.cnn_dense.Weights*reshape(A,64,n)+s.cnn_dense.Bias);
Y = double(s.head.Weights)*double(H)+double(s.head.Bias);
Y = Y+double(s.linear_skip.Weights)*double(X)+double(s.linear_skip.Bias);
end

function Y = conv_same(X,s)
% Two-point same convolution pads the bottom and right edges.
h = size(X,1); w = size(X,2); n = size(X,4);
P = zeros(h+1,w+1,size(X,3),n,'like',X);
P(1:h,1:w,:,:) = X;
Y = zeros(h,w,size(s.Weights,4),n,'like',X);
for out = 1:size(s.Weights,4)
    Z = zeros(h,w,1,n,'like',X);
    for in = 1:size(s.Weights,3)
        for col = 1:2
            for row = 1:2
                Z = Z+P(row:row+h-1,col:col+w-1,in,:).*s.Weights(row,col,in,out);
            end
        end
    end
    Y(:,:,out,:) = Z+s.Bias(out);
end
end

function Y = pool2(X)
Y = (X(1:2:end,1:2:end,:,:)+X(2:2:end,1:2:end,:,:) ...
    +X(1:2:end,2:2:end,:,:)+X(2:2:end,2:2:end,:,:))/cast(4,'like',X);
end

function value = projected_value(H,d)
Hr = H-(H*d.Qa)*d.Qa';
[Q,~,~,keep] = stable_svd(Hr',norm(H,'fro'));
Qr = Q(:,keep);
value = (d.traceG-trace(Qr'*d.G*Qr)+d.floor)/d.normalizer;
end

function text = sci(value)
assert(isscalar(value) && isfinite(value),'DeepONet:NonFiniteValue', ...
    'Cannot report a nonfinite learning rate or loss.');
text = regexprep(sprintf('%.3e',value),'e([+-])0+(\d+)','e$1$2');
end

function atomic(file,s)
temporary = [tempname(fileparts(file)) '.mat'];
cleanup = onCleanup(@()delete_if_present(temporary));
save(temporary,'-struct','s','-v7.3');
[ok,message] = movefile(temporary,file,'f');
assert(ok,'AC:Save','%s',message);
end

function delete_if_present(file)
if isfile(file)
    delete(file);
end
end
function [L, s] = spectrum(G)
	[L, d] = eig((G+G')/2, 'vector');
	[d, order] = sort(real(d), 'descend');
	assert(min(d) >= -1e-10*max(d), 'Invalid negative Gram eigenvalue.');
	L = L(:, order);
	s = sqrt(max(d, 0));
end

function [F, r] = filter_slice(A, threshold)
	[L, S, ~] = svd(A, 'econ');
	r = energy_rank(diag(S), threshold);
	F = L(:, 1:r)*(L(:, 1:r)'*A);
end

function c = project_cache(f, Q, R, q, variance, temporal)
	n = size(f, 'u', 2);
	ns = 1;
	if temporal
		ns = q.nspace;
	end
	C = zeros(ns*size(Q, 2), n);
	energy = 0;
	for first = 1:32:n
		ids = first:min(first+31, n);
		Y = double(f.u(:, ids));
		if temporal
			C(:, ids) = project_time(Q, Y, q.nspace, q.nt);
		else
			C(:, ids) = Q'*Y;
		end
		energy = energy+sum(Y.^2, 'all');
	end
	c = struct('target', C, 'R', R, 'ns', ns, 'floor', max(0, energy-sum(C.^2, 'all')), ...
		'normalizer', q.nspace*q.nt*n*variance, 'scale', ones(size(C, 1), 1));
end

function [mu, scale] = coefficient_statistics(C, alpha)
	mu = single(mean(C, 2));
	scale = ones(size(mu), 'single');
	if alpha == 1
		scale = single(std(C, 0, 2));
		scale(scale <= max(1e-12, 1e-12*max(scale))) = 1;
	end
end

function r = energy_rank(s, threshold)
	r = find(cumsum(s.^2) >= threshold*sum(s.^2), 1);
	assert(~isempty(r));
end

function C = project_time(B, Y, nx, nt)
	C = reshape(permute(reshape(reshape(permute(reshape(Y, nx, nt, []), [2 1 3]), nt, [])'*B, nx, [], size(B, ...
		2)), [1 3 2]), nx*size(B, 2), []);
end

function Y = decode_time(B, C, nx, nt)
	n = size(C, 2);
	r = size(B, 2);
	Y = reshape(permute(reshape(B*reshape(permute(reshape(C, nx, r, n), [2 1 3]), r, []), nt, nx, n), [2 1 3]), ...
		nx*nt, n);
end


function net = initialize_affine(net, X, C, R, mu, scale, cutoff)
	% Fit the affine readout using training data only.
	D = [double(X); ones(1, size(X, 2))];
	physical = solve_geometry(C, R);
	theta = ((physical-double(mu))./double(scale))*pinverse(D, cutoff);
	for j = 1:height(net.Learnables)
		if strcmp(net.Learnables.Layer{j}, 'linear_skip')
			if strcmp(net.Learnables.Parameter{j}, 'Weights')
				v = theta(:, 1:end-1);
			else
				v = theta(:, end);
			end
			net.Learnables.Value{j} = dlarray(single(v));
		end
	end
end


function [f, g] = coefficient_loss(b, X, c)
	f = coefficient_value(b, X, c, []);
	g = dlgradient(f, b.Learnables);
end

function f = coefficient_value(b, X, c, ~)
	C = c.scale.*stripdims(forward(b, X))+c.mu;
	E = map_geometry(C, c)-c.target;
	f = (sum(E.^2, 'all')+c.floor)/c.normalizer;
end


function H = features(net, X)
	H = [];
	for first = 1:64:size(X, 2)
		ids = first:min(first+63, size(X, 2));
		v = forward(net, dlarray(cast(X(:, ids), 'like', extractdata(net.Learnables.Value{1})), 'CB'), ...
			Outputs = 'features');
		a = double(gather(extractdata(v)));
		if isempty(H)
			H = zeros(size(a, 1), size(X, 2));
		end
		H(:, ids) = a; %#ok<AGROW>
	end
end

function net = refit(net, X, C, R, mu, cs, useGPU)
	H = features(net, X);
	physical = solve_geometry(C, R);
	theta = stable_readout(X, H, (physical-double(mu))./double(cs));
	net = set_readout(net, theta, size(X, 1), size(H, 1), zeros(size(mu)), useGPU);
end

function n = set_readout(n, theta, m, h, mu, useGPU)
	for j = 1:height(n.Learnables)
		layer = string(n.Learnables.Layer(j));
		par = string(n.Learnables.Parameter(j));
		if layer == "linear_skip"
			if par == "Weights"
				a = theta(:, 1:m);
			else
				a = theta(:, end)-double(mu);
			end
		elseif layer == "head"
			if par == "Weights"
				a = theta(:, m+(1:h));
			else
				a = zeros(size(theta, 1), 1);
			end
		else
			continue;
		end
		n.Learnables.Value{j} = dlarray(devicearray(single(a), useGPU));
	end
end

function n = feature_net(net)
	ix = find(string({net.Layers.Name}) == "features");
	n = dlnetwork(net.Layers(1:ix));
	for j = 1:height(n.Learnables)
		k = (string(net.Learnables.Layer) == string(n.Learnables.Layer(j))&string(net.Learnables.Parameter) == string(n.Learnables.Parameter(j)));
		n.Learnables.Value{j} = dlarray(double(gather(extractdata(net.Learnables.Value{k}))));
	end
end

function n = assemble(template, feature, X, C, R, mu, cs, useGPU)
	n = template;
	for j = 1:height(feature.Learnables)
		k = (string(n.Learnables.Layer) == string(feature.Learnables.Layer(j))&string(n.Learnables.Parameter) == string(feature.Learnables.Parameter(j)));
		n.Learnables.Value{k} = dlarray(single(gather(extractdata(feature.Learnables.Value{j}))));
	end
	n = netmove(n, @(a)devicearray(a, useGPU));
	n = refit(n, X, C, R, mu, cs, useGPU);
end

function info = readout_diagnostics(b, X, C, R, mu, cs, c)
	% Audit the stored FP32 representation with FP64 readout accumulation before accessing test truth.
	b = netmove(b, @gather);
	H = features(b, X);
	[~, info, Qa, Qh] = stable_readout(X, H, zeros(1, size(X, 2)));
	ideal = (double(C)*Qa)*Qa'+(double(C)*Qh)*Qh';
	s = numeric_net(b);
	C32 = double(single(mu))+double(single(cs)).*numeric_forward(s, single(X));
	for layer = fieldnames(s)'
		for parameter = fieldnames(s.(layer{1}))'
			s.(layer{1}).(parameter{1}) = double(s.(layer{1}).(parameter{1}));
		end
	end
	C64 = double(single(mu))+double(single(cs)).*numeric_forward(s, double(X));
	P32 = map_geometry(double(C32), R);
	P64 = map_geometry(C64, R);
	info.projected_training_mse = (sum((double(C)-ideal).^2, 'all')+c.floor)/c.normalizer;
	info.fp32_training_mse = (sum((double(C)-P32).^2, 'all')+c.floor)/c.normalizer;
	info.fp32_projection_relative_difference = norm(P32-ideal, 'fro')/max(norm(ideal, 'fro'), eps);
	info.fp64_arithmetic_relative_difference = norm(P64-P32, 'fro')/max(norm(P64, 'fro'), eps);
	weights = [];
	for layer = {'head', 'linear_skip'}
		weights = [weights; s.(layer{1}).Weights(:); s.(layer{1}).Bias(:)]; %#ok<AGROW>
	end
	info.max_abs_readout = max(abs(weights));
	info.readout_frobenius = norm(weights);
	assert(info.fp64_arithmetic_relative_difference < 2e-5, 'AC:ArithmeticStability', ...
		'Mixed/full FP64 export arithmetic discrepancy %.9g exceeds 2e-5; check the readout conditioning.', ...
		info.fp64_arithmetic_relative_difference);
end


function policy = numerical_policy()
	% FP32-aware spectral truncation policy.
	policy = struct('version', 'fp32_rank_v2', 'relative_cutoff', sqrt(double(eps('single'))), ...
		'reference', 'unprojected_frobenius', 'input', 'stored_fp32_normalization', ...
		'readout', 'affine_first_truncated_svd', 'vp_gradient', 'spectral_projector_exact');
end

function [U, s, V, keep, threshold] = stable_svd(A, reference)
	% QR reduces the factorization while preserving the original singular values.
	A = double(A);
	if nargin < 2
		reference = norm(A, 'fro');
	end
	if size(A, 1) >= size(A, 2)
		[Q, R] = qr(A, 0);
		[U, S, V] = svd(R, 'econ');
		U = Q*U;
	else
		[U, S, V] = svd(A, 'econ');
	end
	s = diag(S);
	policy = numerical_policy();
	threshold = max(policy.relative_cutoff*reference, max(size(A))*eps('double')*reference);
	keep = s > threshold;
end

function [Qa, Pa, threshold] = affine_space(X)
	D = [double(X); ones(1, size(X, 2))];
	[U, s, V, keep, threshold] = stable_svd(D');
	Qa = U(:, keep);
	Pa = (Qa./s(keep)')*V(:, keep)';
end

function [theta, info, Qa, Qh] = stable_readout(X, H, C)
	% Resolve the affine span first, then fit only its orthogonal complement.
	[Qa, Pa, affineThreshold] = affine_space(X);
	Hr = double(H)-(double(H)*Qa)*Qa';
	[U, s, V, keep, hiddenThreshold] = stable_svd(Hr', norm(double(H), 'fro'));
	Qh = U(:, keep);
	B = ((double(C)*Qh)./s(keep)')*V(:, keep)';
	A = (double(C)-B*double(H))*Pa;
	theta = [A(:, 1:end-1), B, A(:, end)];
	info = struct('affine_rank', size(Qa, 2), 'hidden_rank', nnz(keep), ...
		'affine_threshold', affineThreshold, 'hidden_threshold', hiddenThreshold, ...
		'max_abs_readout', max(abs(theta), [], 'all'), 'readout_frobenius', norm(theta, 'fro'));
end

function d = vp_prepare(X, C, floorSSE, normalizer, ~)
	[Qa, ~, threshold] = affine_space(X);
	Cr = double(C)-(double(C)*Qa)*Qa';
	G = Cr'*Cr;
	G = (G+G')/2;
	d = struct('Qa', Qa, 'G', G, 'traceG', sum(Cr.^2, 'all'), 'floor', double(floorSSE), ...
		'normalizer', double(normalizer), 'affine_threshold', threshold);
end

function [value, J, info] = projected_objective(H, d)
	% Differentiate the retained spectral projector, including nonzero discarded modes.
	Hr = H-(H*d.Qa)*d.Qa';
	[Q, s, V, keep, threshold] = stable_svd(Hr', norm(H, 'fro'));
	r = find(keep);
	z = find(~keep);
	Qr = Q(:, r);
	Vr = V(:, r);
	GQ = Q'*d.G;
	cross = GQ*Q;
	value = (d.traceG-trace(cross(r, r))+d.floor)/d.normalizer;
	J = -2*(Vr./s(r)')*(GQ(r, :)-cross(r, :)*Q');
	if ~isempty(r) && ~isempty(z)
		gap = s(r).^2-(s(z)').^2;
		coupling = cross(r, z)./gap;
		J = J-2*(Vr*(s(r).*coupling)*Q(:, z)'+V(:, z)*(s(z).*coupling')*Qr');
	end
	J = (J-(J*d.Qa)*d.Qa')/d.normalizer;
	info = struct('hidden_rank', numel(r), 'hidden_threshold', threshold);
end

function [f, g] = vp_loss(net, X, d)
	Hdl = forward(net, X);
	H = extractdata(stripdims(Hdl));
	[value, J] = projected_objective(H, d);
	surrogate = sum(stripdims(Hdl).*J, 'all');
	g = dlgradient(surrogate, net.Learnables);
	f = dlarray(value);
end

function P = pinverse(A, ~)
	[U, s, V, keep] = stable_svd(A);
	P = (V(:, keep)./s(keep)')*U(:, keep)';
end


function v = pack(varargin)
	parts = {};
	for k = 1:nargin
		tab = varargin{k};
		if isempty(tab)
			continue;
		end
		if isa(tab, 'dlnetwork')
			tab = tab.Learnables;
		end
		for j = 1:height(tab)
			parts{end+1} = reshape(tab.Value{j}, [], 1); %#ok<AGROW>
		end
	end
	v = cat(1, parts{:});
end

function [b, t] = unpack(v, b, t)
	offset = 0;
	for j = 1:height(b.Learnables)
		n = numel(b.Learnables.Value{j});
		b.Learnables.Value{j} = reshape(v(offset+(1:n)), size(b.Learnables.Value{j}));
		offset = offset+n;
	end
	if ~isempty(t)
		for j = 1:height(t.Learnables)
			n = numel(t.Learnables.Value{j});
			t.Learnables.Value{j} = reshape(v(offset+(1:n)), size(t.Learnables.Value{j}));
			offset = offset+n;
		end
	end
	assert(offset == numel(v));
end

function n = netmove(n, fun)
	if isempty(n)
		return;
	end
	for j = 1:height(n.Learnables)
		n.Learnables.Value{j} = dlarray(fun(extractdata(n.Learnables.Value{j})));
	end
end

function c = move_fields(c, gpu, fp32)
	if nargin < 3
		fp32 = true;
	end
	for f = fieldnames(c)'
		v = c.(f{1});
		if isnumeric(v) && ~strcmp(f{1}, 'ns')
			if fp32
				v = single(v);
			end
			v = devicearray(v, gpu);
			if fp32
				v = dlarray(v);
			end
			c.(f{1}) = v;
		end
	end
end


function a = devicearray(a, gpu)
	if gpu
		a = gpuArray(a);
	else
		a = gather(a);
	end
end


function syncgpu(yes)
	if yes
		wait(gpuDevice);
	end
end

function y = number(x)
	if isa(x, 'dlarray')
		x = extractdata(x);
	end
	y = double(gather(x));
end

function s = numeric_net(n)
	s = struct;
	if isempty(n)
		return;
	end
	for j = 1:height(n.Learnables)
		s.(char(n.Learnables.Layer(j))).(char(n.Learnables.Parameter(j))) = single(gather(extractdata(n.Learnables.Value{j})));
	end
end


function C = map_geometry(C, c)
	if c.ns == 1
		C = c.R*C;
	else
		n = size(C, 2);
		r = size(c.R, 2);
		V = reshape(permute(reshape(C, c.ns, r, n), [2 1 3]), r, []);
		C = reshape(permute(reshape(c.R*V, size(c.R, 1), c.ns, n), [2 1 3]), [], n);
	end
end

function C = solve_geometry(C, c)
	if ~isstruct(c)
		C = pinverse(c, 0)*double(C);
	elseif c.ns == 1
		C = pinverse(c.R, 0)*double(C);
	else
		n = size(C, 2);
		r = size(c.R, 1);
		V = reshape(permute(reshape(C, c.ns, r, n), [2 1 3]), r, []);
		C = reshape(permute(reshape(pinverse(c.R, 0)*double(V), size(c.R, 2), c.ns, n), [2 1 3]), [], n);
	end
end


function n = numeric_count(s)
	n = 0;
	for f = fieldnames(s)'
		a = s.(f{1});
		for g = fieldnames(a)'
			n = n+numel(a.(g{1}));
		end
	end
end


function X = normalize_input(X, n)
	X = (single(X)-n.mean)./(n.std+n.epsilon);
end

function Y = native_export_forward(net, X)
	% Retain native FP32 hidden features and accumulate both output channels in FP64.
	X = gather(extractdata(stripdims(X)));
	H = features(net, X);
	s = numeric_net(net);
	Y = double(s.head.Weights)*H+double(s.head.Bias);
	if isfield(s, 'linear_skip')
		Y = Y+double(s.linear_skip.Weights)*double(X)+double(s.linear_skip.Bias);
	end
end







