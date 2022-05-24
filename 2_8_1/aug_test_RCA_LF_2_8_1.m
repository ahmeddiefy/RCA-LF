clearvars; clearvars -global; clc; close all; warning off;
dataset = 'occ';
sceneFolder = strcat('./data/TrainingData/Test/',dataset,'/');

extension = [];
list = dir([sceneFolder, '/*', extension]);
sceneNames = setdiff({list.name}, {'.', '..'});
scenePaths = strcat(strcat(sceneFolder, '/'), sceneNames);
numScenes = length(sceneNames);


for ns = 1:numScenes

    numImgsX = 14;
    numImgsY = 14;
    
    resultPath = [scenePaths{ns}];
    sceneName = sceneNames{ns};
    sceneName = sceneName(1:end-4);
%%% converting the extracted light field to a different format
    inputImg = im2double(imread(resultPath));
    inputImg = rgb2ycbcr(inputImg);

    h = size(inputImg, 1) / numImgsY;
    w = size(inputImg, 2) / numImgsX;
    
    fullLF = zeros(h, w, 3, numImgsY, numImgsX);
    
    for ax = 1 : numImgsX
        for ay = 1 : numImgsY
            fullLF(:, :, :, ay, ax) = inputImg(ay:numImgsY:end, ax:numImgsX:end, :);
        end
    end

    fullLF = fullLF(1:h, 1:w, :, 4:11, 4:11); % we only take the 8 middle images
    
    input_small = zeros(2,2,h,w,3);
    input_small_1 = zeros(2,8,h,w,3);
    image_ycbcr = zeros(8,8,h,w,3);

    
    input_small(1,1,:,:,:) = fullLF(:,:,:,2,2);
    input_small(1,2,:,:,:) = fullLF(:,:,:,2,7);
    
    input_small(2,1,:,:,:) = fullLF(:,:,:,7,2);
    input_small(2,2,:,:,:) = fullLF(:,:,:,7,7);

   
    
     for i = 1:h
        for v=1:2
          input_small_1(v,:,i,:,:) = imresize(squeeze(input_small(v,:,i,:,:)),[8 w]);   
        end
    end
    
    for i = 1:w
        for v=1:8
          image_ycbcr(:,v,:,i,:) = imresize(squeeze(input_small_1(:,v,:,i,:)),[8 h]);   
        end
    end
    
    
   image_ycbcr=single(permute(image_ycbcr,[3,4,5,1,2]));
      
    
    
    
    image_ycbcr(:,:,:,2,2) = fullLF(:,:,:,2,2);
    image_ycbcr(:,:,:,2,7) = fullLF(:,:,:,2,7);
    
    image_ycbcr(:,:,:,7,2) = fullLF(:,:,:,7,2);
    image_ycbcr(:,:,:,7,7) = fullLF(:,:,:,7,7);

    
    gt = zeros(h, w, 3, 64);
    in = zeros(h, w, 3, 64);
    for ax = 1 : 8
            for ay = 1 : 8
                gt( :, :, :, sub2ind([8 8], ax, ay)) = fullLF(:, :, :, ay, ax);
                in( :, :, :, sub2ind([8 8], ax, ay)) = image_ycbcr(:, :, :, ay, ax);
            end
     end
    
    
    
    
    
    gt = single(gt);
    in = single(in);
    
        
    patch_name = strcat('./data/testLF/',dataset,'/gt/',sceneName);
    save(patch_name, 'gt');
    patch_name = strcat('./data/testLF/',dataset,'/in/',sceneName);
    save(patch_name, 'in');
    
    

    
    display(ns);
    
    
end
