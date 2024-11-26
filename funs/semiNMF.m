function [W, H, dnorm] = semiNMF(X, k,iter)
    % 初始化矩阵 H
    H = rand(k, size(X, 2));
    
    % 设置参数
    max_iter = iter;
    tolfun = 1e-5;
    
    % 初始化矩阵 W
    W = rand(size(X, 1), k);
    
    % 计算初始偏差
    dnorm = norm(X - W * H, 'fro');
    
    for i = 1:max_iter
        % 更新矩阵 WW
        W = W .* ((X * H') ./ (W * H * H' + eps));
        
        % 更新矩阵 H
        H = H .* ((W' * X) ./ (W' * W * H + eps));
        
        % 计算偏差
        new_dnorm = norm(X - W * H, 'fro');
        disp(new_dnorm);
        % 检查是否满足收敛条件
        if abs(dnorm - new_dnorm) <= tolfun * max(1, dnorm)
            disp('final max_iter');
            disp(i);
            break;
        end
        if i== max_iter
            disp('final max_iter');
            disp(i);
            break;
        end
        dnorm = new_dnorm;
    end
end
