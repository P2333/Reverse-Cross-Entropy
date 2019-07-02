clc
clear all
labels=load('kernel_para_cifar10/training_logitslabels_RCE');
logits=load('kernel_para_cifar10/training_logits_RCE');
num_per_class=1000;

kernel_RCE=zeros(num_per_class,64,10);

for i=1:10
    train_logits=logits(find(labels==(i-1)),:);
    index=randsample(1:size(train_logits,1),num_per_class);
    kernel_RCE(:,:,i)=train_logits(index,:);
end
save('kernel_para_cifar10/kernel1000_RCE.mat','kernel_RCE')