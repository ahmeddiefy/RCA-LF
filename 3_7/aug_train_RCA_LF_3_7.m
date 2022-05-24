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

    fullLF = fullLF(1:h, 1:w, 5:11, 5:11); % we only take the 7 middle images
    



   
    input_small = zeros(h,w,3,3);
    input_small_1 = zeros(h,w,3,7);
    input_small_2 = zeros(h,w,7,7);

    
    input_small(:,:,1,1) = fullLF(:,:,1,1);
    input_small(:,:,1,2) = fullLF(:,:,1,4);
    input_small(:,:,1,3) = fullLF(:,:,1,7);
    
    input_small(:,:,2,1) = fullLF(:,:,4,1);
    input_small(:,:,2,2) = fullLF(:,:,4,4);
    input_small(:,:,2,3) = fullLF(:,:,4,7);
    
    input_small(:,:,3,1) = fullLF(:,:,7,1);
    input_small(:,:,3,2) = fullLF(:,:,7,4);
    input_small(:,:,3,3) = fullLF(:,:,7,7);
   
    
    for i = 1:h
        for v=1:3
          input_small_1(i,:,v,:) = imresize(squeeze(input_small(i,:,v,:)),[w 7]);   
        end
    end
    
    for i = 1:w
        for v=1:7
          input_small_2(:,i,:,v) = imresize(squeeze(input_small_1(:,i,:,v)),[h 7]);   
        end
    end
    
    
    
    
    input_small_2(:,:,1,1) = fullLF(:,:,1,1);
    input_small_2(:,:,1,4) = fullLF(:,:,1,4);
    input_small_2(:,:,1,7) = fullLF(:,:,1,7);
    
    input_small_2(:,:,4,1) = fullLF(:,:,4,1);
    input_small_2(:,:,4,4) = fullLF(:,:,4,4);
    input_small_2(:,:,4,7) = fullLF(:,:,4,7);
    
    input_small_2(:,:,7,1) = fullLF(:,:,7,1);
    input_small_2(:,:,7,4) = fullLF(:,:,7,4);
    input_small_2(:,:,7,7) = fullLF(:,:,7,7);
    
    img_raw = zeros(h, w, 49);
    img_1 = zeros(h, w, 49);
    for ax = 1 : 7
            for ay = 1 : 7
                img_raw( :, :, sub2ind([7 7], ax, ay)) = fullLF(:, :, ay, ax);
                img_1( :, :, sub2ind([7 7], ax, ay)) = input_small_2(:, :, ay, ax);
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