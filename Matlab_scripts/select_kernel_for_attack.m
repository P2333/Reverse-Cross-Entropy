clc
clear all
num_per_class=1000;
num_per_class_for_attack=100;

load('kernel_para_mnist/kernel1000_RCE.mat')
kernel_RCE_for_attack=zeros(num_per_class_for_attack,64,10);

for i=1:10
    index=randsample(1:num_per_class,num_per_class_for_attack);
    kernel_RCE_for_attack(:,:,i)=kernel_RCE(index,:,i);
end
save('kernel_para_mnist/kernel1000_for_attack_RCE.mat','kernel_RCE_for_attack')