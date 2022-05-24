clearvars; clearvars -global; clc; close all; warning off;
sceneFolder = './data/TrainingData/Training';
patch_size = 32;
count = 1;
extension = [];
list = dir([sceneFolder, '/*', extension]);
sceneNames = setdiff({list.name}, {'.', '..'});
scenePaths = strcat(strcat(sceneFolder, '/'), sceneNames);
numScenes = length(sceneNames);

for ns = 1:numScenes

    numImgsX = 14;
    numImgsY = 14;
    
    resultPath = [scenePaths{ns}];
    
%%% converting the extracted light field to a different format
    inputImg = im2double(imread(resultPath))
    inputImg = rgb2ycbcr(inputImg);
    inputImg = inputImg(:,:,1);

    h = size(inputImg, 1) / numImgsY;
    w = size(inputImg, 2) / numImgsX;
    
    fullLF = zeros(h, w, numImgsY, numImgsX);
    
    for ax = 1 : numImgsX
        for ay = 1 : numImgsY
            fullLF(:, :, ay, ax) = inputImg(ay:numImgsY:end, ax:numImgsX:end);
        end
    end

    fullLF = fullLF(1:h, 1:w, 4:11, 4:11); % we only take the 8 middle images
    



   
    input_small = zeros(h,w,2,2);
    input_small_1 = zeros(h,w,2,8);
    input_small_2 = zeros(h,w,8,8);

    
    input_small(:,:,1,1) = fullLF(:,:,1,1);
    input_small(:,:,1,2) = fullLF(:,:,1,8);
    
    input_small(:,:,2,1) = fullLF(:,:,8,1);
    input_small(:,:,2,2) = fullLF(:,:,8,8);

   
    
    for i = 1:h
        for v=1:2
          input_small_1(i,:,v,:) = imresize(squeeze(input_small(i,:,v,:)),[w 8]);   
        end
    end
    
    for i = 1:w
        for v=1:8
          input_small_2(:,i,:,v) = imresize(squeeze(input_small_1(:,i,:,v)),[h 8]);   
        end
    end
    
    
    
    
    input_small_2(:,:,1,1) = fullLF(:,:,1,1);
    input_small_2(:,:,1,8) = fullLF(:,:,1,8);

    input_small_2(:,:,8,1) = fullLF(:,:,8,1);
    input_small_2(:,:,8,8) = fullLF(:,:,8,8);

    
    img_raw = zeros(h, w, 64);
    img_1 = zeros(h, w, 64);
    for ax = 1 : 8
            for ay = 1 : 8
                img_raw( :, :, sub2ind([8 8], ax, ay)) = fullLF(:, :, ay, ax);
                img_1( :, :, sub2ind([8 8], ax, ay)) = input_small_2(:, :, ay, ax);
            end
     end
    
    
    
    
    
    img_raw = single(img_raw);
    img_1 = single(img_1);
    
    [H,W,~] = size(img_raw);
        
    for ix=1:floor(H/patch_size)
        for iy=1:floor(W/patch_size)
           patch_name = sprintf('./data/train/%d',count);
           img_raw_patch =  img_raw( (ix-1)*patch_size + 1:ix * patch_size, (iy-1)*patch_size + 1:iy * patch_size,:);
           img_1_patch= img_1( (ix-1)*patch_size + 1:ix * patch_size, (iy-1)*patch_size + 1:iy * patch_size,:);
           patch = img_raw_patch;
           save(patch_name, 'patch');
           patch = img_1_patch;
           save(sprintf('%s_1', patch_name), 'patch');
           count = count+1;
        end
    end

    
    display(ns);
    
    
end