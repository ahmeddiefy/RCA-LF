clearvars; clearvars -global; clc; close all; warning off;
dataset = '30s';
sceneFolder = strcat('./data/TrainingData/Test/',dataset);
count = 0;

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

    fullLF = fullLF(1:h, 1:w, :, 5:11, 5:11); % we only take the 7 middle images
    
    input_small = zeros(3,3,h,w,3);
    input_small_1 = zeros(3,7,h,w,3);
    image_ycbcr = zeros(7,7,h,w,3);

    
    input_small(1,1,:,:,:) = fullLF(:,:,:,1,1);
    input_small(1,2,:,:,:) = fullLF(:,:,:,1,4);
    input_small(1,3,:,:,:) = fullLF(:,:,:,1,7);
    
    input_small(2,1,:,:,:) = fullLF(:,:,:,4,1);
    input_small(2,2,:,:,:) = fullLF(:,:,:,4,4);
    input_small(2,3,:,:,:) = fullLF(:,:,:,4,7);

    input_small(3,1,:,:,:) = fullLF(:,:,:,7,1);
    input_small(3,2,:,:,:) = fullLF(:,:,:,7,4);
    input_small(3,3,:,:,:) = fullLF(:,:,:,7,7);
   
    
     for i = 1:h
        for v=1:3
          input_small_1(v,:,i,:,:) = imresize(squeeze(input_small(v,:,i,:,:)),[7 w]);   
        end
    end
    
    for i = 1:w
        for v=1:7
          image_ycbcr(:,v,:,i,:) = imresize(squeeze(input_small_1(:,v,:,i,:)),[7 h]);   
        end
    end
    
    
   image_ycbcr=single(permute(image_ycbcr,[3,4,5,1,2]));
      
    
    
    
    image_ycbcr(:,:,:,1,1) = fullLF(:,:,:,1,1);
    image_ycbcr(:,:,:,1,4) = fullLF(:,:,:,1,4);
    image_ycbcr(:,:,:,1,7) = fullLF(:,:,:,1,7);
    
    image_ycbcr(:,:,:,4,1) = fullLF(:,:,:,4,1);
    image_ycbcr(:,:,:,4,4) = fullLF(:,:,:,4,4);
    image_ycbcr(:,:,:,4,7) = fullLF(:,:,:,4,7);
    
    image_ycbcr(:,:,:,7,1) = fullLF(:,:,:,7,1);
    image_ycbcr(:,:,:,7,4) = fullLF(:,:,:,7,4);
    image_ycbcr(:,:,:,7,7) = fullLF(:,:,:,7,7);
    
    gt = zeros(h, w, 3, 49);
    in = zeros(h, w, 3, 49);
    for ax = 1 : 7
            for ay = 1 : 7
                gt( :, :, :, sub2ind([7 7], ax, ay)) = fullLF(:, :, :, ay, ax);
                in( :, :, :, sub2ind([7 7], ax, ay)) = image_ycbcr(:, :, :, ay, ax);
            end
     end
    
    
    
    
    
    gt = single(gt);
    in = single(in);
    
        
    patch_name = strcat('./data/testLF/',dataset,'/gt/',sceneName);
    save(patch_name, 'gt');
    patch_name = strcat('./data/testLF/',dataset,'/in/',sceneName);
    save(patch_name, 'in');
    count = count+1;
    
    

    
    display(count);
    
    
end
