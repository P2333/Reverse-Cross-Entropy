clc
clear all
dataset='cifar10';
f1='RCE';
num_per_class=100;


labels=load(['kernel_para_',dataset,'/training_logitslabels_',f1]);
logits=load(['kernel_para_',dataset,'/training_logits_',f1]);
load(['kernel_para_',dataset,'/kernel1000_for_attack_',f1,'.mat'])


median_out=zeros(10,1);
for i=1:10
    index=find(labels==(i-1));
    s=size(index,1);
    if strcmp(f1,'CE')
        sigma2=1/0.26;
        kernel_vec=kernel_CE_for_attack(:,:,i);
    else
        sigma2=0.1/0.26;
        kernel_vec=kernel_RCE_for_attack(:,:,i);
    end
    density=zeros(s,1);
    for j=1:s
        logits_one=logits(index(j),:);
        density(j)=mean(exp(-sum((repmat(logits_one,num_per_class,1)-kernel_vec).^2,2)/sigma2));
    end
    median_out(i)=median(density);
end
save(['kernel_para_',dataset,'/kernel1000_median_for_attack_',f1,'.mat'],'median_out')