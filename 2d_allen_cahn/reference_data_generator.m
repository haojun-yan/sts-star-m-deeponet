function data = reference_data_generator(cfg,work_dir)
% Generate shared FP32 trajectories on a periodic square grid.
r = cfg.reference_data;
rng(r.data_seed,'twister');
c = struct('N_x',r.Nx,'N_y',r.Ny,'N_s',r.Nx*r.Ny,'N_t',r.Nt, ...
    'x',(0:r.Nx-1)'/r.Nx,'y',(0:r.Ny-1)'/r.Ny,'t',linspace(0,1,r.Nt)', ...
    'kappa',r.kappa,'initial_modes',[1 0;0 1;1 1;1 -1]);
[sx,sy] = ndgrid((0:15)/16);
sensor_coordinates = [sx(:),sy(:)];
counts = [r.num_of_training_sample,r.num_of_validation_sample,r.num_of_testing_sample];
directions = randn(8,sum(counts));
parameters = directions./sqrt(sum(directions.^2,1)).*rand(1,sum(counts)).^(1/8);
rawsensors = initial_values(sensor_coordinates,parameters,c);
data = struct('x',c.x,'y',c.y,'t',c.t,'sensor_coordinates',sensor_coordinates, ...
    'grid_shape',[r.Nx,r.Ny,r.Nt],'output_layout','x fastest, then y, then time; columns are cases');
fprintf('[Data] Training: %d | Validation: %d | Test: %d\n',counts);
fprintf('[Data] Grid: Nx %d, Ny %d, Nt %d, sensors %d, reference %d\n',r.Nx,r.Ny,r.Nt,r.nsensor,r.Nref);
fprintf('[Data] Solver: Fourier ETDRK4, doubled-grid cubic dealiasing, dt %.9g\n',r.dt);
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
    s.reference_configuration = c;
    s.reference_configuration.Nref = r.Nref;
    s.reference_configuration.dt = r.dt;
    save(file,'-struct','s','-v7.3');
    f = matfile(file,'Writable',true);
    f.u(r.Nx*r.Ny*r.Nt,counts(j)) = single(0);
    for first = 1:r.batch_size:counts(j)
        localids = first:min(first+r.batch_size-1,counts(j));
        fields = zeros(r.Nx*r.Ny*r.Nt,numel(localids),'single');
        batch_parameters = parameters(:,ids(localids));
        Nref = r.Nref;
        dt = r.dt;
        parfor k = 1:numel(localids)
            a = reference_trajectory(batch_parameters(:,k),c,Nref,dt);
            fields(:,k) = single(reshape(permute(a,[1 3 2]),[],1));
        end
        assert(all(isfinite(fields),'all'),'AC:Reference','Nonfinite reference trajectory.');
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

function u = initial_values(xy, z, cfg)
	phase = 2*pi*xy*cfg.initial_modes';
	nmodes = size(cfg.initial_modes, 1);
	u = .25*(cos(phase)*z(1:nmodes, :)+sin(phase)*z(nmodes+1:end, :));
end

function output = reference_trajectory(z, cfg, N, h)
	% Fourier ETDRK4 uses a doubled grid for exact retained cubic convolution.
	P = 2*N;
	k = [0:N/2-1, -N/2:-1];
	[kx, ky] = ndgrid(k);
	L = 1-cfg.kappa*(2*pi)^2*(kx.^2+ky.^2);
	E = exp(h*L);
	E2 = exp(h*L/2);
	roots = exp(1i*pi*((1:32)-.5)/32);
	LR = h*L+reshape(roots, 1, 1, []);
	Q = h*real(mean((exp(LR/2)-1)./LR, 3));
	f1 = h*real(mean((-4-LR+exp(LR).*(4-3*LR+LR.^2))./LR.^3, 3));
	f2 = h*real(mean((2+LR+exp(LR).*(-2+LR))./LR.^3, 3));
	f3 = h*real(mean((-4-3*LR-LR.^2+exp(LR).*(4-LR))./LR.^3, 3));
	[x, y] = ndgrid((0:N-1)/N);
	u = reshape(initial_values([x(:), y(:)], z, cfg), N, N, []);
	v = fft2(u)/N^2;
	v(N/2+1, :, :) = 0;
	v(:, N/2+1, :) = 0;
	output = zeros(cfg.N_s, size(z, 2), cfg.N_t);
	stride = N/cfg.N_x;
	assert(stride == fix(stride));
	steps = round(cfg.t(end)/h);
	outputdt = cfg.t(2)-cfg.t(1);
	saveevery = round(outputdt/h);
	assert(saveevery >= 1 && abs(saveevery*h-outputdt) < 1e-14);
	assert(max(abs(cfg.t-round(cfg.t/h)*h)) < 1e-12);
	it = 1;
	for step = 0:steps
		if mod(step, saveevery) == 0
			u = real(ifft2(v)*N^2);
			output(:, :, it) = reshape(u(1:stride:end, 1:stride:end, :), cfg.N_s, []);
			it = it+1;
		end
		if step == steps
			break;
		end
		Nv = cubic_nonlinearity(v, N, P);
		a = E2.*v+Q.*Nv;
		Na = cubic_nonlinearity(a, N, P);
		b = E2.*v+Q.*Na;
		Nb = cubic_nonlinearity(b, N, P);
		c = E2.*a+Q.*(2*Nb-Nv);
		Nc = cubic_nonlinearity(c, N, P);
		v = E.*v+f1.*Nv+2*f2.*(Na+Nb)+f3.*Nc;
		v(N/2+1, :, :) = 0;
		v(:, N/2+1, :) = 0;
	end
	assert(it == cfg.N_t+1, 'Reference output recording count differs.');
end

function Nv = cubic_nonlinearity(v, N, P)
	idx = [1:N/2, P-N/2+1:P];
	padded = zeros(P, P, size(v, 3), 'like', v);
	padded(idx, idx, :) = v;
	u = real(ifft2(padded)*P^2);
	w = fft2(-u.^3)/P^2;
	Nv = w(idx, idx, :);
	Nv(N/2+1, :, :) = 0;
	Nv(:, N/2+1, :) = 0;
end

