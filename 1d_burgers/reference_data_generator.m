function data = reference_data_generator(cfg, work_dir)
% Generate shared FP32 trajectories with conditioned Gaussian initial fields.
r = cfg.reference_data;
rng(r.data_seed,'twister');
c = struct('N_x',r.Nx,'N_t',r.Nt,'x',linspace(-1,1,r.Nx)', ...
    't',linspace(0,1,r.Nt)','sensor_coordinates',linspace(-1,1,r.nsensor)', ...
    'nu',r.nu,'sigma',r.sigma,'length_scale',r.length_scale, ...
    'covariance_jitter',r.covariance_jitter,'quadrature_points',r.quadrature_points, ...
    'image_radius',r.image_radius,'initial_interpolation','pchip', ...
    'sensor_interpolation','linear','solver','Cole--Hopf reflected Neumann heat kernel', ...
    'initial_family','zero-boundary conditioned Gaussian random field', ...
    'pde',sprintf('u_t + u*u_x = %.9g*u_xx; u(-1,t)=u(1,t)=0',r.nu), ...
    'protocol_metadata',struct('reference_N',512,'reference_dt',2.5e-4, ...
        'used_by_active_solver',false));
counts = [r.num_of_training_sample,r.num_of_validation_sample,r.num_of_testing_sample];
K = c.sigma^2*exp(-(c.x-c.x').^2/(2*c.length_scale^2));
boundary = [1,r.Nx];
Kb = K(:,boundary);
L = chol(K-Kb*(K(boundary,boundary)\Kb')+c.covariance_jitter*eye(r.Nx),'lower');
parameters = L*randn(r.Nx,sum(counts));
parameters(boundary,:) = 0;
rawsensors = interp1(c.x,parameters,c.sensor_coordinates);
data = struct('x',c.x,'t',c.t,'sensor_coordinates',c.sensor_coordinates, ...
    'grid_shape',[r.Nx,r.Nt],'output_layout','space fastest, then time; columns are cases', ...
    'reference_configuration',c);
fprintf('[Data] Training: %d | Validation: %d | Test: %d\n',counts);
fprintf('[Data] Grid: Nx %d, Nt %d, sensors %d, quadrature %d\n',r.Nx,r.Nt,r.nsensor,r.quadrature_points);
fprintf('[Data] Solver: Cole--Hopf, PCHIP, reflected Neumann kernel, image radius %d\n',r.image_radius);
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
        fields = burgers_reference(parameters(:,ids(localids)),c);
        fields = single(reshape(permute(fields,[1 3 2]),r.Nx*r.Nt,[]));
        assert(all(isfinite(fields),'all'),'Burgers:Reference','Nonfinite reference trajectory.');
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

function fields = burgers_reference(initial,c)
pp = cell(size(initial,2),1);
for j = 1:numel(pp)
    pp{j} = pchip(c.x,initial(:,j));
end
y = -1+((1:c.quadrature_points)'-.5)*(2/c.quadrature_points);
phi = make_phi(pp,y,c.nu);
fields = heat_reference(phi,y,c.x,c.t,c.nu,initial,c.image_radius);
end

function phi = make_phi(pp,y,nu)
phi = zeros(numel(y),numel(pp));
for j = 1:numel(pp)
    a = -primitive_pp(pp{j},y)/(2*nu);
    a = a-max(a);
    phi(:,j) = exp(a);
end
end

function v = primitive_pp(pp,x)
[breaks,c,pieces,order,dim] = unmkpp(pp);
assert(dim == 1);
h = diff(breaks)';
ic = [c./(order:-1:1),zeros(pieces,1)];
areas = zeros(pieces,1);
for j = 1:order
    areas = areas+c(:,j).*h.^(order-j+1)/(order-j+1);
end
ic(:,end) = [0;cumsum(areas(1:end-1))];
v = ppval(mkpp(breaks,ic),x);
v = v(:);
end

function fields = heat_reference(phi,y,x,t,nu,initial,radius)
% Reflect the Neumann heat kernel before the Cole--Hopf transformation.
fields = zeros(numel(x),size(phi,2),numel(t));
fields(:,:,1) = initial;
D = x-y';
S = x+y'+2;
parfor it = 2:numel(t)
    K = zeros(size(D));
    J = zeros(size(D));
    for n = -radius:radius
        for form = 1:2
            if form == 1
                d = D+4*n;
            else
                d = S+4*n;
            end
            G = exp(-d.^2/(4*nu*t(it)));
            K = K+G;
            J = J+d.*G;
        end
    end
    denominator = K*phi;
    numerator = J*phi;
    assert(all(denominator > 0,'all'),'Burgers:Reference','Nonpositive Cole--Hopf denominator.');
    u = numerator./denominator/t(it);
    u([1 end],:) = 0;
    fields(:,:,it) = u;
end
end
