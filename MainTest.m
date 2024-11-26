clear all
clc

addpath([pwd, '/funs']);
addpath([pwd, '/datasets']);
% datasetName = 'HW';
% anchorRate = [1.0];
% p = [0.01];
% lambda = [10];
% %% load Dataset
datasetName = 'BBCSport_unshuffled';
anchorRate = [1.0];
p = [0.8];
lambda = [20];
load(['./datasets/',datasetName, '.mat']);
Y = labels;
X = data;
for i = 1:length(X)
    X{i} = X{i}';
end
gt = Y;
num_Cluster = length(unique(gt));               
num_V = length(X);                              
num_N = size(X{1},1);                           
% prerocessing = 'None';
%% Data preprocessing
% Select a data preprocessing method, or no data preprocessing
% MSRC:     None 
% HW1256:   Data pre-processing B
% Mnist4:   Data pre-processing A
% AWA:      Data pre-processing A
% Data pre-processing A
disp('------Data preprocessing------');
prerocessing = 'preA';
tic
for v=1:num_V
    a = max(X{v}(:));
    X{v} = double(X{v}./a);
end
toc

% % Data pre-processing B
% disp('------Data preprocessing------');
% prerocessing = 'preB';
% tic
% for v=1:num_V
%     XX = X{v};
% for n=1:size(XX,1)
%     XX(n,:) = XX(n,:)./norm(XX(n,:),'fro');
% end
% X{v} = double(XX);
% end
% toc


%% parameter
% MSRC:     anchorRate:0.7 p:0.5 lambda:100
% HW1256:   anchorRate:1.0 p:0.1 lambda:1180
% Mnist4:   anchorRate:0.6 p:0.1 lambda:5000
% AWA:      anchorRate:1.0 p:0.5 lambda:1000
% anchorRate = [0.5,0.7,0.8,1.0];
% p = [0.05,0.1,0.2,0.3,0.5];
% % lambda = [10,50,100,200,500,1000];
% anchorRate = [0.6];
% p = [0.15];
% lambda = [100];

anchorNum = fix(num_N * anchorRate);
layers_L = [10,5,1];
for num1 = 1:length(anchorNum)
    fprintf('------Current Anchor number:%d------\n', anchorNum(num1));

    %% result file
    dir_name = ['.\result\', datasetName, '\'];
    file_dir = [dir_name, datasetName,'_with_', prerocessing,'_with_weighting_with_', int2str(anchorNum(num1)), 'AnchorPoints.csv'];
    if ~isfolder(dir_name)
        mkdir(dir_name);
    end
    fid = fopen(file_dir,'a');  

    %% 
    disp('----------Anchor Selection----------');
    tic;
    opt1.style = 1;          
    opt1.IterMax = 50;                      
    opt1.toy = 0;

    [~, B_init] = FastmultiCLR(X, num_Cluster, anchorNum(num1), opt1, 10);    
    toc;

    %% 
    B_init_hat = time2frequency(B_init);
    for v = 1:num_V 
        F_init_hat{v} = eye(num_N, num_Cluster);
        G_init_hat{v} = B_init_hat{v}' * F_init_hat{v};
    end    
    NMF_iter = [750];
    maxIter = [120];
    %% 
    for num2 = 1:length(p)
        for num3 = 1:length(lambda)
            % NMF_iter = 500;
            % maxIter = 180;
            for num4 = 1:length(NMF_iter)
                for num5 = 1:length(maxIter)
                    % [F, alpha] = OrthNTF(num_N, num_V, num_Cluster, B_init_hat, F_init_hat, G_init_hat, p(num2), lambda(num3),maxIter(num5));
                    % F_sum = F{1} / alpha(1);
                    % for v = 2:num_V
                    %     F_sum = F_sum + F{v} / alpha(v);
                    % end
                    % alpha_sum = sum(1 ./ alpha);
                    % F_final = F_sum / alpha_sum;
                    % [~, Y_pre] = max(F_final, [], 2); 
                    % my_result = ClusteringMeasure1(Y, Y_pre);
                    % my_result;
                    % % %% result file
                    % % dir_name = ['.\result\', datasetName, '\'];
                    % % file_dir = [dir_name, datasetName, '_with_', int2str(anchorNum(num1)), 'AnchorPoints.csv'];
                    % % if ~isfolder(dir_name)
                    % %     mkdir(dir_name);
                    % % end
                    % % fid = fopen(file_dir,'a');  
                    % % fprintf('myorth : %g %g %g %g %g %g %g \n',my_result');
                    % %% 
                    % fprintf(fid, '  ORTH  :Current Anchor number::,%f, ', anchorNum(num1));
                    % fprintf(fid, 'maxIter::,%f, ',maxIter(num5) );
                    % fprintf(fid, 'NMF_iter::,%f, ',NMF_iter(num4) );
                    % % fprintf(fid, 'switch::%f ',orth );
                    % fprintf(fid, 'P:,%f, ', p(num2));
                    % fprintf(fid, 'lambda:,%f, ', lambda(num3));
                    % fprintf(fid, '  orth  :, %g,%g,%g,%g,%g,%g,%g \n',my_result');  

                    [F, alpha] = myOrthNTF(datasetName,gt,layers_L, num_N, num_V, num_Cluster, B_init_hat, F_init_hat, G_init_hat, p(num2), lambda(num3),maxIter(num5),NMF_iter(num4));
                    F_sum = F{1} / alpha(1);
                    for v = 2:num_V
                        F_sum = F_sum + F{v} / alpha(v);
                    end
                    alpha_sum = sum(1 ./ alpha);
                    F_final = F_sum / alpha_sum;
                    [~, Y_pre] = max(F_final, [], 2); 
                    my_result = ClusteringMeasure1(Y, Y_pre);
                    my_result;
                    fprintf('myorth : %g %g %g %g %g %g %g \n',my_result');
                    fprintf(fid, 'myOrth  :Current Anchor number::,%f, ', anchorNum(num1));
                    fprintf(fid, 'maxIter::,%f, ',maxIter(num5) );
                    fprintf(fid, 'NMF_iter::,%f, ',NMF_iter(num4) );
                    % fprintf(fid, 'switch::%f ',orth );
                    fprintf(fid, 'P:,%f, ', p(num2));
                    fprintf(fid, 'lambda:,%f, ', lambda(num3));
                    fprintf(fid, 'myOrth  :, %g,%g,%g,%g,%g,%g,%g \n',my_result');  
                    % [F, alpha] = myOrthNTF3(layers_L,num_N, num_V, num_Cluster, B_init_hat, F_init_hat, G_init_hat, p(num2), lambda(num3),maxIter(num5),NMF_iter(num4));
                    % F_sum = F{1} / alpha(1);
                    % for v = 2:num_V
                    %     F_sum = F_sum + F{v} / alpha(v);
                    % end
                    % alpha_sum = sum(1 ./ alpha);
                    % F_final = F_sum / alpha_sum;
                    % [~, Y_pre] = max(F_final, [], 2); 
                    % my_result = ClusteringMeasure1(Y, Y_pre);
                    % my_result;
                    % % result file
                    % dir_name = ['.\result\', datasetName, '\'];
                    % file_dir = [dir_name, datasetName, '_with_', int2str(anchorNum(num1)), 'AnchorPoints.csv'];
                    % if ~isfolder(dir_name)
                    %     mkdir(dir_name);
                    % end
                    % fid = fopen(file_dir,'a');  
                    % 
                    % %% 
                    % fprintf(fid, 'myOrthSp  :Current Anchor number::,%f, ', anchorNum(num1));
                    % fprintf(fid, 'maxIter::,%f, ',maxIter(num5) );
                    % fprintf(fid, 'NMF_iter::,%f, ',NMF_iter(num4) );
                    % % fprintf(fid, 'switch::%f ',orth );
                    % fprintf(fid, 'P:,%f, ', p(num2));
                    % fprintf(fid, 'lambda:,%f, ', lambda(num3));
                    % fprintf(fid, 'myOrthSp  :, %g,%g,%g,%g,%g,%g,%g \n',my_result'); 
                    
                end
            end
        end
    end
    fclose(fid);
end
