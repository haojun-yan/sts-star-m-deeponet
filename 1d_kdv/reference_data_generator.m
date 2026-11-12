function data = reference_data_generator(cfg, work_dir)
% Generate shared FP32 trajectories on a fixed periodic grid.
r = cfg.reference_data;
rng(r.data_seed, 'twister');
c = struct('N_x', r.Nx, 'N_t', r.Nt, 'x', 2*pi*(0:r.Nx-1)'/r.Nx, ...
    't', linspace(0,1,r.Nt)', 'sigma', r.sigma, 'initial_modes', (1:6)');
c.initial_weights = (1+c.initial_modes.^2/9).^-1;
c.initial_S = sum(c.initial_weights.^2);
sensor_coordinates = 2*pi*(0:r.nsensor-1)'/r.nsensor;
counts = [r.num_of_training_sample,r.num_of_validation_sample,r.num_of_testing_sample];
parameters = randn(12,sum(counts));
rawsensors = kdv_initial(sensor_coordinates,parameters,c);
data = struct('x',c.x,'t',c.t,'sensor_coordinates',sensor_coordinates, ...
    'grid_shape',[r.Nx,r.Nt],'output_layout','space fastest, then time; columns are cases');
fprintf('[Data] Training: %d | Validation: %d | Test: %d\n',counts);
fprintf('[Data] Grid: Nx %d, Nt %d, sensors %d, reference %d\n',r.Nx,r.Nt,r.nsensor,r.Nref);
fprintf('[Data] Solver: Fourier ETDRK4, three-halves padding, dt %.9g\n',r.dt);
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
    s.initialparameters = parameters(:,ids);
    save(file,'-struct','s','-v7.3');
    f = matfile(file,'Writable',true);
    f.u(r.Nx*r.Nt,counts(j)) = single(0);
    for first = 1:r.batch_size:counts(j)
        localids = first:min(first+r.batch_size-1,counts(j));
        fields = zeros(r.Nx*r.Nt,numel(localids),'single');
        batch_parameters = parameters(:,ids(localids));
        Nref = r.Nref;
        dt = r.dt;
        parfor k = 1:numel(localids)
            a = kdv_reference(batch_parameters(:,k),c,Nref,dt);
            fields(:,k) = single(reshape(permute(a,[1 3 2]),[],1));
        end
        assert(all(isfinite(fields),'all'),'KdV:Reference','Nonfinite reference trajectory.');
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
function u = kdv_initial(x, z, cfg)
	k = cfg.initial_modes;
	w = cfg.sigma/sqrt(cfg.initial_S)*cfg.initial_weights;
	u = (cos(x*k').*w')*z(1:6, :)+(sin(x*k').*w')*z(7:12, :);
end

function fields = kdv_reference(z, cfg, N, dt)
	nsteps = round(1/dt);
	dt = 1/nsteps;
	savesteps = round(cfg.t*nsteps);
	assert(max(abs(savesteps/nsteps-cfg.t)) < 1e-12, 'Output times must be reference steps.');
	assert(mod(N, cfg.N_x) == 0, 'Reference and output grids must nest.');
	x = 2*pi*(0:N-1)'/N;
	k = [0:N/2-1, 0, -N/2+1:-1]';
	L = 1i*k.^3;
	E = exp(dt*L);
	E2 = exp(dt*L/2);
	% Full-circle quadrature preserves complex coefficients for the imaginary linear operator.
	roots = exp(2i*pi*((1:64)-.5)/64);
	Z = dt*L+roots;
	Q = dt*mean((exp(Z/2)-1)./Z, 2);
	f1 = dt*mean((-4-Z+exp(Z).*(4-3*Z+Z.^2))./Z.^3, 2);
	f2 = dt*mean((2+Z+exp(Z).*(-2+Z))./Z.^3, 2);
	f3 = dt*mean((-4-3*Z-Z.^2+exp(Z).*(4-Z))./Z.^3, 2);
	n = size(z, 2);
	fields = zeros(cfg.N_x, n, cfg.N_t);
	for first = 1:64:n
		ids = first:min(first+63, n);
		v = fft(kdv_initial(x, z(:, ids), cfg));
		v(N/2+1, :) = 0;
		u = real(ifft(v));
		fields(:, ids, 1) = u(1:N/cfg.N_x:end, :);
		frame = 2;
		for step = 1:nsteps
			Nv = kdv_nonlinear(v, k);
			a = E2.*v+Q.*Nv;
			Na = kdv_nonlinear(a, k);
			b = E2.*v+Q.*Na;
			Nb = kdv_nonlinear(b, k);
			c = E2.*a+Q.*(2*Nb-Nv);
			Nc = kdv_nonlinear(c, k);
			v = E.*v+f1.*Nv+2*f2.*(Na+Nb)+f3.*Nc;
			if frame <= cfg.N_t && step == savesteps(frame)
				u = real(ifft(v));
				fields(:, ids, frame) = u(1:N/cfg.N_x:end, :);
				frame = frame+1;
			end
		end
		assert(frame == cfg.N_t+1, 'Reference output recording count differs.');
	end
end

function Nv = kdv_nonlinear(v, k)
	% Three-halves padding evaluates the retained quadratic convolution without aliasing.
	N = size(v, 1);
	P = 3*N/2;
	padded = zeros(P, size(v, 2), 'like', v);
	padded(1:N/2, :) = v(1:N/2, :)*P/N;
	padded(P-N/2+2:P, :) = v(N/2+2:N, :)*P/N;
	u = real(ifft(padded));
	square = fft(u.^2)*N/P;
	truncated = [square(1:N/2, :); zeros(1, size(v, 2)); square(P-N/2+2:P, :)];
	Nv = -3i*k.*truncated;
end

