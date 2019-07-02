clc 
clear all
%attack_method_all={'fgsm','bim','tgsm','jsma','carliniL2','carliniL2_highcon'};
attack_method_all={'bim'};
num_per_eps=1000;
dataset='mnist';
eps_round=8; %eps=0.02 * eps_round for iterative-based attacks

num_attack_method=size(attack_method_all,2);
auc=zeros(6,num_attack_method);
%% Loading parameters
%Loading training parameters
for count=1:num_attack_method
attack_method=attack_method_all{count};

load(['kernel_para_',dataset,'/kernel1000_RCE.mat'])

num_per_class=size(kernel_RCE,1);
%Loading RCE results
logits_RCE_adv_all=load([attack_method,'_',dataset,'/RCE/logits_adv']);
logits_RCE_nor_all=load([attack_method,'_',dataset,'/RCE/logits_nor']);

labels_RCE_adv_all=reshape(load([attack_method,'_',dataset,'/RCE/labels_adv']),num_per_eps,[]);
labels_RCE_nor_all=reshape(load([attack_method,'_',dataset,'/RCE/labels_nor']),num_per_eps,[]);
labels_RCE_true_all=reshape(load([attack_method,'_',dataset,'/RCE/labels_true']),num_per_eps,[]);

confidence_RCE_adv_all=reshape(load([attack_method,'_',dataset,'/RCE/confidence_adv']),num_per_eps,[]);
entropy_RCE_adv_all=reshape(load([attack_method,'_',dataset,'/RCE/entropy_adv']),num_per_eps,[]);
confidence_RCE_nor_all=reshape(load([attack_method,'_',dataset,'/RCE/confidence_nor']),num_per_eps,[]);
entropy_RCE_nor_all=reshape(load([attack_method,'_',dataset,'/RCE/entropy_nor']),num_per_eps,[]);

%% Controling hyperparameters
yita_RCE=1;

sigma2_RCE=0.1/0.26;
entropy_yita_RCE=10^-4;

%% Calculate density
id_range=(1+num_per_eps*(eps_round-1)):(num_per_eps+num_per_eps*(eps_round-1));

%RCE, choose the samples that correctly classified as nor img and wrongly classified as adv img
labels_RCE_adv=labels_RCE_adv_all(:,eps_round);%1000X1
labels_RCE_nor=labels_RCE_nor_all(:,eps_round);%1000X1
labels_RCE_true=labels_RCE_true_all(:,eps_round);%1000X1
if 1==0
    correct_nor_and_succ_adv_RCE=(1:num_per_eps)';
else
    correct_nor_and_succ_adv_RCE=find(labels_RCE_adv~=labels_RCE_true&labels_RCE_nor==labels_RCE_true);
end
num_correct_RCE=size(correct_nor_and_succ_adv_RCE,1);
id_RCE=id_range(correct_nor_and_succ_adv_RCE);
labels_RCE_adv=labels_RCE_adv_all(correct_nor_and_succ_adv_RCE,eps_round);%1000X1
labels_RCE_nor=labels_RCE_nor_all(correct_nor_and_succ_adv_RCE,eps_round);%1000X1
labels_RCE_true=labels_RCE_true_all(correct_nor_and_succ_adv_RCE,eps_round);%1000X1
logits_RCE_adv=logits_RCE_adv_all(id_RCE,:);%1000X64
logits_RCE_nor=logits_RCE_nor_all(id_RCE,:);%1000X64
entropy_RCE_adv=entropy_RCE_adv_all(correct_nor_and_succ_adv_RCE,eps_round);%1000X1
entropy_RCE_nor=entropy_RCE_nor_all(correct_nor_and_succ_adv_RCE,eps_round);%1000X1
confidence_RCE_adv=confidence_RCE_adv_all(correct_nor_and_succ_adv_RCE,eps_round);%1000X1
confidence_RCE_nor=confidence_RCE_nor_all(correct_nor_and_succ_adv_RCE,eps_round);%1000X1


%Density
density_RCE_nor=zeros(num_correct_RCE,1);
density_RCE_adv=zeros(num_correct_RCE,1);

for i=1:num_correct_RCE
    kernel_vec_nor=kernel_RCE(:,:,labels_RCE_nor(i)+1);
    kernel_vec_adv=kernel_RCE(:,:,labels_RCE_adv(i)+1);
    density_RCE_nor(i,1)=mean(exp(-sum((repmat(logits_RCE_nor(i,:),num_per_class,1)-kernel_vec_nor).^2,2)/sigma2_RCE));
    density_RCE_adv(i,1)=mean(exp(-sum((repmat(logits_RCE_adv(i,:),num_per_class,1)-kernel_vec_adv).^2,2)/sigma2_RCE));
end
%% Calculate auc and plot roc
targets_RCE=[ones(1,num_correct_RCE) zeros(1,num_correct_RCE)];

%3 metrics for RCE
outputs_RCE_den=[density_RCE_nor' density_RCE_adv'];
outputs_RCE_con=[confidence_RCE_nor' confidence_RCE_adv'];
outputs_RCE_nonME=[entropy_RCE_nor' entropy_RCE_adv'];

%calculate auc for RCE
auc_RCE_con=AUC(targets_RCE, outputs_RCE_con);
auc(4,count)=auc_RCE_con;
auc_RCE_den=AUC(targets_RCE, outputs_RCE_den);
auc(5,count)=auc_RCE_den;
auc_RCE_nonME=AUC(targets_RCE, outputs_RCE_nonME);
auc(6,count)=auc_RCE_nonME;
end

%Output auc-roc scores
auc_RCE_con
auc_RCE_nonME
auc_RCE_den