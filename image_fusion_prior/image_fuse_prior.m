% image fuse prior：
clear

x1_suf = ".png"; 
x2_suf = x1_suf;

f1_suf = x1_suf;
f2_suf = x1_suf;
f3_suf = x1_suf;

x1_path = "C:\Users\我本飞扬\Desktop\集合\test_img\ir_vis_MSRS\part1_three_channel"; % source image pairs
x2_path = "C:\Users\我本飞扬\Desktop\集合\test_img\ir_vis_MSRS\part2_initial";

f1_path = 'C:\Users\我本飞扬\Desktop\集合\各方法测试结果\Diff-IF\ir_vis_MSRS\'; % iamge fuse prior 1
f2_path = 'C:\Users\我本飞扬\Desktop\集合\各方法测试结果\SwinFusion\ir_vis_MSRS\'; % iamge fuse prior 2
f3_path = 'C:\Users\我本飞扬\Desktop\集合\各方法测试结果\U2Fusion\ir_vis_MSRS\'; % iamge fuse prior 3

prior_path(1).name=f1_path;
prior_path(2).name=f2_path;
prior_path(3).name=f3_path;

x1_list = dir(x1_path + "\*" + x1_suf);
x2_list = dir(x2_path + "\*" + x2_suf);

f1_list = dir(f1_path + "\*" + f1_suf);
f2_list = dir(f2_path + "\*" + f2_suf);
f3_list = dir(f3_path + "\*" + f3_suf);

prior_list(:,1)=f1_list;
prior_list(:,2)=f2_list;
prior_list(:,3)=f3_list;

img_num = size(x1_list,1); % image num
prior_num=3;  % prior num
metrics_num=6; % metrics num

prior(1).name='Diff-IF'; 
prior(2).name='SwinFusion'; 
prior(3).name='U2Fusion'; 

metrics(1).name='SF';
metrics(2).name='EN';
metrics(3).name='SSIM';
metrics(4).name='FMI';
metrics(5).name='Qabf';
metrics(6).name='VIF';

fuse_path = 'C:\Users\我本飞扬\Desktop\集合\fusion_prior';
record=zeros(1,20);
for i = 1:img_num
    
    fprintf('正在针对各融合先验处理第%d幅图像....\n',i);
    
    x1 = imread(char(x1_path + "\" + string(x1_list(i).name)));
    x2 = imread(char(x2_path + "\" + string(x2_list(i).name)));
    
    for j= 1:prior_num  
       
       fprintf('计算第%d个融合先验的指标度量...\n', j);       
       f_list = prior_list(:,j);
       fuse = imread(strcat(prior_path(j).name, f_list(i).name));  
       result = eval(x1,x2,fuse);
       table(:,j) = result;
         
    end
    
    fprintf('准备输出各融合先验各指标及名次,各列依次是：\n  Diff-IF，SwinFsuion, U2Fusion...\n'); 
    
    for j= 1:metrics_num
       fprintf('%s：\n', metrics(j).name);
       disp(table(j,:));
       table_result(j,:) = tiedrank(-table(j,:)); 
       disp(table_result(j,:));
    end
   
    fprintf('融合先验的各指标排序：\n各列依次是：Diff-IF，SwinFsuion，U2Fusion\n各列依次是：SF，EN，SSIM，FMI, Qabf，VIF\n');
    disp(table_result); 

    eval_result= sum(table_result); % 得到各方法各指标的名次
    fprintf('各融合先验的名次：\n');
    disp(eval_result);
    
    [min_value, min_index] = min(eval_result);
    fprintf('累计的最小值是: %d\n', min_value);
    fprintf('最小值所在的位置是: %d\n', min_index);
    fprintf('相应的融合方法是: %s \n', prior(min_index).name);
    f_list = prior_list(:,min_index);   
    fuse = imread(strcat(prior_path(min_index).name, f_list(min_index).name));  
    imwrite(fuse, strcat(fuse_path, '\', f_list(min_index).name));
    fprintf('图像保存成功\n\n\n');
    record(i)=min_index;  
end

fprintf('各图像来源于图像融合先验: %d\n', record);

function  result=eval(img1,img2,fused)
    
    SF= roundn(metrics_SF(img1, img2, fused),-4);
    EN = roundn(metrics_EN(img1,img2,fused),-4);   
    SSIM = roundn(metrics_SSIM(img1,img2,fused),-4);
    FMI = roundn(metrics_FMI(img1, img2, fused),-4);  
    Qabf = roundn(metrics_Qabf(img1, img2, fused),-4);
    VIF = roundn((metrics_VIF(img1, fused) + metrics_VIF(img2, fused)),-4);
    result=[SF;EN;SSIM;FMI;Qabf;VIF];
end



