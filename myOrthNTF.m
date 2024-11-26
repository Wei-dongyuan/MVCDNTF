function [Q, alpha] = myOrthNTF(datasetName,gt,layers_L,num_N, num_V, num_C, B_init,  F_init, G_init, p, beta, maxIter,NMF_iter)
if (~exist('maxIter', 'var'))
    maxIter = 150;
end
filename = sprintf('%s_results.txt', datasetName);

% Open a file for writing
fid = fopen(filename, 'w');
if fid == -1
    error('Cannot open file for writing: %s', filename);
end
% 默认参数
z0 = 0;
h0 = 0;
bUpdateH = 1;
bUpdateLastH = 1;
maxiter = 150;
tolfun = 1e-6;
verbose = 0;
bUpdateZ = 1;
cache = 0;
gnd = 0;
lambda = 0;
savePath = '';  % 新添加的参数
layers = num_C * layers_L;
%% initial
num_Anchor = size(B_init{1}, 2);
sX = [num_N, num_C, num_V];
Isconverg = 0;
iter = 0;
eta = 1.3;
mu = 10e-5;
rho = 10e-5;
max_mu = 10e12;
max_rho = 10e12;
betaf = ones(num_V, 1); 

for v = 1:num_V
    Y1{v} = zeros(num_N, num_C);           
    J{v} = zeros(num_N, num_C);             
    Y2{v} = zeros(num_N, num_C);
    Q{v} = zeros(num_N, num_C);
end
numOfLayer = numel(layers);
S_hat = B_init;
H_hat = F_init;
G_hat = G_init;
Z_hat_layers = cell(num_V, numOfLayer);
H_hat_layers = cell(num_V, numOfLayer);
alpha = repmat(1 / num_V, [1,num_V]);       
timeStart = clock;
for v_ind = 1:num_V
    if  ~iscell(h0)
        for i_layer = 1:length(layers)
            if i_layer == 1
                % For the first layer we go linear from X to Z*H, so we use id
                V = S_hat{v_ind};
                Z_hat_layers{v_ind,i_layer} = eye(layers(i_layer),num_N);
            else
                V = H_hat_layers{v_ind,i_layer-1}';
            end
            [Z_hat_layers{v_ind,i_layer}, H_hat_layers{v_ind,i_layer}, ~] = semiNMF(V',layers(i_layer),NMF_iter);
            % H_hat_layers{v_ind,i_layer} = eye(layers(i_layer),num_N);
            % Z_hat_layers{v_ind,i_layer} = V' * H_hat_layers{v_ind,i_layer};
            % Z_hat_layers{v_ind,i_layer} = eye(layers(i_layer),num_N);
            % F_init_hat{v} = eye(num_N, num_Cluster);
            % G_init_hat{v} = B_init_hat{v}' * F_init_hat{v};
            % if i_layer ==1 
            %     mDz{v_ind} = Z_hat_layers{v_ind,i_layer};
            % else
            %     mDz{v_ind} = mDz{v_ind} * Z_hat_layers{v_ind,i_layer};
            % end
            % if i_layer == length(layers)
            %    H_hat{v_ind} = H_hat_layers{v_ind,i_layer}';
            %    G_hat{v_ind} = mDz{v_ind};
            % end
            % if ~iscell(z0)
            %     % For the later layers we use nonlinearities as we go from
            %     % g(H_{k-1}) to Z*H_k
            %     [Z{v_ind,i_layer}, H{v_ind,i_layer}, ~] = ...
            %         seminmf(V, ...
            %         layers(i_layer), ...
            %         'maxiter', maxiter, ...
            %         'bUpdateH', true, 'bUpdateZ', bUpdateZ, 'verbose', verbose, 'save', cache, 'fast', 1);
            % else
            %     disp('Using existing Z');
            %     [Z{v_ind,i_layer}, H{v_ind,i_layer}, ~] = ...
            %         seminmf(V, ...
            %         layers(i_layer), ...
            %         'maxiter', 1, ...
            % %         'bUpdateH', true, 'bUpdateZ', 0, 'z0', z0{i_layer}, 'verbose', verbose, 'save', cache, 'fast', 1);
            % end
        end
    else
        Z_hat_layers=z0;
        H_hat_layers=h0;
        if verbose
            disp('Skipping initialization, using provided init matrices...');
        end
    end
end

converg = [];
listSumNormHQ = [];
listSumNormHJ = [];
while(Isconverg == 0)  
    % %% update G_hat{v}
    % % for v = 1:num_V
    % %     G_hat{v} = S_hat{v}' * H_hat{v};
    % % end
    % G = frequency2time(G_hat);
    %% update G_hat{v}
    for v_ind = 1:num_V
        for i = 1:numOfLayer
            H_err{v_ind,numOfLayer} = H_hat{v_ind}';
            for i_layer = numOfLayer-1:-1:1
                H_err{v_ind,i_layer} = Z_hat_layers{v_ind,i_layer+1} * H_err{v_ind,i_layer+1};
            end
            %update Zi
            if bUpdateZ
                try
                    if i == 1
                        Z_hat_layers{v_ind,i} = S_hat{v_ind}'  * pinv(H_err{v_ind,1});
                    else
                        Z_hat_layers{v_ind,i} = pinv(Dz) * S_hat{v_ind}' * pinv(H_err{v_ind,i});
                    end
                catch
                    fprintf('Convergance error %f. min Z{i}: %f. max %f\n', norm(Z{v_ind,i}, 'fro'), min(min(Z{v_ind,i})), max(max(Z{v_ind,i})));
                end
            end
            if i == 1
                Dz = Z_hat_layers{v_ind,1};
            else
                Dz = Dz * Z_hat_layers{v_ind,i};
            end

            % update Hi
            if bUpdateH && (i < numOfLayer) || (i==numOfLayer) && bUpdateLastH          
                % DT*X -> DTX
                DTX =  Dz' * S_hat{v_ind}';
                DTXp = (abs(DTX)+DTX)./2;
                DTXn = (abs(DTX)-DTX)./2;

                % DT*D -> DTD
                DTD =  Dz' * Dz;
                DTDp = (abs(DTD)+DTD)./2;
                DTDn = (abs(DTD)-DTD)./2;

                H_hat_layers{v_ind,i} = H_hat_layers{v_ind,i} .* sqrt((DTXp + DTDn * H_hat_layers{v_ind,i}) ./ max(DTXn + DTDp * H_hat_layers{v_ind,i}, 1e-10));
                if i == numOfLayer
                    H_hat_layers{v_ind,i} = H_hat{v_ind}';
                end
            end

        end
        G_hat{v_ind} = Dz;
    end
    % G = frequency2time(G_hat);
    %% update H_hat{v}
    for v = 1:num_V
        temp1_hat{v} = S_hat{v} * G_hat{v};
    end
    temp1 = frequency2time(temp1_hat);
    for v = 1:num_V  
        B{v} = 2*temp1{v} + mu*Q{v} - Y1{v} + rho*J{v} - Y2{v}; 
    end
    B_hat = time2frequency(B);
    for v = 1:num_V
        [nn{v}, ~, vv{v}] = svd(B_hat{v}, 'econ');
        H_hat{v} = nn{v} * vv{v}';
    end
    H = frequency2time(H_hat);
    clear B nn vv;    
    
    %% update Q_hat{v}
    for v = 1:num_V
        Q{v} = H{v} + Y1{v} ./ mu;
        Q{v}(Q{v}<0) = 0;
    end
    Q_hat = time2frequency(Q);
        
    %% update tensor J{v}
    for v = 1:num_V
        M{v} = H{v} + Y2{v} ./ rho;
    end
    M_tensor = cat(3, M{:,:});
    M_vector = M_tensor(:);
    [myj, ~] = wshrinkObj_weight_lp(M_vector, beta*betaf./rho, sX, 1, 3, p);
    J_tensor = reshape(myj, sX);
    for v = 1:num_V
        J{v} = J_tensor(:,:,v);
    end
    J_hat = time2frequency(J);
    clear M M_tensor M_vector M_tensor;
    
    %% update Y1 Y2 mu rho
    for v = 1:num_V
        Y1{v} = Y1{v} + mu * (H{v} - Q{v});
        Y2{v} = Y2{v} + rho * (H{v} - J{v});
    end
    mu = min(eta * mu, max_mu);
    rho = min(eta * rho, max_rho);        
    if ismember(iter, [0, 10, 60,62,63,64,65,66,67,68,70, 90])
        % 使用提供的方法计算F_final
        F_sum = Q{1} ;
        for v = 2:num_V
            F_sum = F_sum + Q{v} ;
        end
        alpha_sum = sum(1 ./ alpha);
        F_final = F_sum ;
        F_final_normalized = zscore(F_final);

        % % 执行 PCA
        % [coeff, score, ~, ~, explained] = pca(real(F_final_normalized));
        % 
        % % 选择前两个主成分用于可视化
        % pc1 = score(:,1);
        % pc2 = score(:,2);
        % 
        % % 可视化结果
        % figure;
        % gscatter(pc1, pc2, gt);
        % title('Iteration %d:PCA Visualization', iter);
        % xlabel(['PC 1 (' num2str(explained(1), '%.2f') '% Variance Explained)']);
        % ylabel(['PC 2 (' num2str(explained(2), '%.2f') '% Variance Explained)']);
        % grid on;
        % [~, Y_pre] = max(F_final, [], 2); 
        % 应用t-SNE进行降维
        F_final_normalized = max(0, min(1, (F_final - min(F_final(:))) / (max(F_final(:)) - min(F_final(:)))));
        % mappedX = tsne(real(F_final_normalized), 'Algorithm', 'exact', 'Perplexity',30, 'LearnRate', 50,  'Distance', 'cosine',  'Standardize', true, 'Verbose', 1);
        mappedX = tsne(real(F_final_normalized));
        % 创建图形进行可视化
        figure;
        gscatter(mappedX(:,1), mappedX(:,2), gt); 
        title(sprintf('Iteration %d: t-SNE Visualization with True Labels', iter));
        xlabel('t-SNE Dimension 1');
        ylabel('t-SNE Dimension 2');

        % 保存图形
        % saveFileName = sprintf('.\\result\\%s\\tSNE_Visualization_Iter_%d_Dim_%d_%d_p_%.2f_lambda_%.2f.png', datasetName, iter, num_N, num_C, p, beta);
        % saveas(gcf, saveFileName);
    end

    if iter > maxIter
        Isconverg = 1;
    end
    % F_sum = Q{1} / alpha(1);
    % for v = 2:num_V
    %     F_sum = F_sum + Q{v} / alpha(v);
    % end
    % alpha_sum = sum(1 ./ alpha);
    % F_final = F_sum / alpha_sum;
    % [~, Y_pre] = max(F_final, [], 2); 
    % my_result = ClusteringMeasure1(gt, Y_pre);
    % fprintf('myorth : %g %g %g %g %g %g %g \n', my_result');
    % % Calculate and store the differences between H and Q, and H and J
    % diff_H_Q = 0;
    % diff_H_J = 0;
    % 
    % % Loop over each view
    % for v = 1:num_V
    %     % Calculate the Frobenius norm of the difference for each view and sum it
    %     diff_H_Q = diff_H_Q + norm(H{v} - Q{v}, 'fro')^2;
    %     diff_H_J = diff_H_J + norm(H{v} - J{v}, 'fro')^2;
    % end

    % % Take the square root of the summed squares to get the overall norm
    % diff_H_Q = sqrt(diff_H_Q);
    % diff_H_J = sqrt(diff_H_J);
    % fprintf(fid, '%d,%f,%f,%g,%g,%g,%g,%g,%g,%g\n', iter, diff_H_Q, diff_H_J, my_result);
    iter = iter + 1;
    fprintf('Final iter:%d\n', iter);
    
end

timeEnd = clock;
fprintf('Time all:%f s\n', etime(timeEnd, timeStart));
fprintf('Time average:%f s\n', etime(timeEnd, timeStart)/iter);
fprintf('Final iter:%d\n', iter);