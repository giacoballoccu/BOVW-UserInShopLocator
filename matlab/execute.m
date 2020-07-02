



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  ICPR 2014 Tutorial                                                     %
%  Hands on Advanced Bag-of-Words Models for Visual Recognition           %
%                                                                         %
%  Instructors:                                                           %
%  L. Ballan     <lamberto.ballan@unifi.it>                               %
%  L. Seidenari  <lorenzo.seidenari@unifi.it>                             %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%   BOW pipeline: Image classification using bag-of-features              %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%   Part 1:  Load and quantize pre-computed image features                %
%   Part 2:  Represent images by histograms of quantized features         %
%   Part 3:  Classify images with nearest neighbor classifi             %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;
close all;

% DATASET
dataset_dir='/train_set/split_by_class_RGB';

% FEATURES extraction methods
% 'sift' for sparse features detection (SIFT descriptors computed at  
% Harris-Laplace keypoints) or 'dsift' for dense features detection (SIFT
% descriptors computed at a grid of overlapped patches

%desc_name = 'sift';
desc_name = 'dsift';
%desc_name = 'msdsift';

% FLAGS
do_create_folders_class = 0
do_convert_input = 0
do_feat_extraction_test = 1;
do_feat_extraction_train = 0;
do_split_sets_train = 0;
do_split_sets_test = 1;

do_form_codebook = 1;
do_feat_quantization = 1;

do_L2_NN_classification = 1;
do_chi2_NN_classification = 0;

visualize_feat = 0;
visualize_words = 0;
visualize_confmat = 0;
visualize_res = 0;
have_screen = ~isempty(getenv('DISPLAY'));

% PATHS
basepath = 'C:/Users/Giaco/Desktop/cv_project/actual_project/';
traintxtpath = strcat(basepath, 'img/egocart', '/train_set/train_set.txt');
testtxtpath = strcat(basepath, 'img/egocart', '/test_set/test_set.txt');
wdir = 'C:/Users/Giaco/Desktop/cv_project/actual_project/';
libsvmpath = [ wdir, fullfile('lib','libsvm-3.11','matlab')];
addpath(libsvmpath)

% BOW PARAMETERS
max_km_iters = 50; % maximum number of iterations for k-means
nfeat_codebook = 60000; % number of descriptors used by k-means for the codebook generation
norm_bof_hist = 1;

% number of codewords (i.e. K for the k-means algorithm)
nwords_codebook = 500;

% image file extension
file_ext='jpg';

%Convert input dataset into directory rappresenting the class and move the images in each folder%

%Converting the trainset.txt into a table
trainSet = readtable(traintxtpath,'Delimiter', ' ', 'HeaderLines', 0, 'ReadVariableNames',true );
trainSet = trainSet(:,1:7);
trainSet = table2cell(trainSet);

%Converting the testset.txt into a table
testSet = readtable(testtxtpath,'Delimiter', ' ', 'HeaderLines', 0, 'ReadVariableNames',true );
testSet = testSet(:,1:7);
testSet = table2cell(testSet);

if do_create_folders_class
    for i = 1:16
        mkdir(strcat(basepath, 'img/egocart/train_set/split_by_class_RGB/'), int2str(i));
        mkdir(strcat(basepath, 'img/egocart/test_set/split_by_class_RGB/'), int2str(i));
    end
end

if do_convert_input
    %Training set
    [M, N] = size(trainSet);
    for i = 1 : M
        class = trainSet(i,7);
        file_RGB = strcat(basepath, 'img/egocart', '/train_set/train_RGB/', trainSet(i,1));
        dest = strcat(basepath, 'img/egocart', '/train_set/split_by_class_RGB/', num2str(cell2mat(class)));
        copyfile(string(file_RGB), dest) 
    end
    %Test set
[M, N] = size(testSet);
for i = 1 : M
    class = testSet(i,7);
    file_RGB = strcat(basepath, 'img/egocart', '/test_set/test_RGB/', testSet(i,1));
    dest = strcat(basepath, 'img/egocart', '/test_set/split_by_class_RGB/', num2str(cell2mat(class)));
    copyfile(string(file_RGB), dest) 
end
end

% Create a new dataset split
file_split = 'split.mat';
if do_split_sets_train    
    data_train = create_dataset_split_structure(strcat(basepath, 'img/egocart'), 1 , file_ext);
    save(fullfile(strcat(basepath, 'img/egocart'),'/train_set/split_by_class_RGB/',file_split),'data');
else
    load(fullfile(strcat(basepath, 'img/egocart'),'/train_set/split_by_class_RGB/',file_split));
end

if do_split_sets_test   
    data_train = create_dataset_split_structure(strcat(basepath, 'img/egocart'), 0 , file_ext);
    save(fullfile(strcat(basepath, 'img/egocart'),'/test_set/split_by_class_RGB/',file_split),'data');
else
    load(fullfile(strcat(basepath, 'img/egocart'),'/test_set/split_by_class_RGB/',file_split));
end
classes = {data_train.classname}; % create cell array of class name strings

% Extract SIFT features fon training and test images
if do_feat_extraction_train   
    extract_sift_features(fullfile(basepath,'img/egocart','/train_set/split_by_class_RGB/'),desc_name)    
end
if do_feat_extraction_test  
    extract_sift_features(fullfile(basepath,'img/egocart','/test_set/split_by_class_RGB/'),desc_name)    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% Part 1: quantize pre-computed image features %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Load pre-computed SIFT features for training images

% The resulting structure array 'desc' will contain one
% entry per images with the following fields:
%  desc(i).r :    Nx1 array with y-coordinates for N SIFT features
%  desc(i).c :    Nx1 array with x-coordinates for N SIFT features
%  desc(i).rad :  Nx1 array with radius for N SIFT features
%  desc(i).sift : Nx128 array with N SIFT descriptors
%  desc(i).imgfname : file name of original image

lasti=1;
for i = 1:length(data)
     images_descs = get_descriptors_files(data,i,file_ext,desc_name,'train');
     for j = 1:length(images_descs) 
        fname = fullfile(basepath,'img/egocart/',dataset_dir,data(i).classname,images_descs{j});
        fprintf('Loading %s /n',fname, '\n');
        tmp = load(fname,'-mat');
        tmp.desc.class=i;
        tmp.desc.imgfname=regexprep(fname,['.' desc_name],'.jpg');
        desc_train(lasti)=tmp.desc;
        desc_train(lasti).sift = single(desc_train(lasti).sift);
        lasti=lasti+1;
     end
end


%% Visualize SIFT features for training images
if (visualize_feat && have_screen)
    nti=10;
    fprintf('/nVisualize features for %d training images/n', nti);
    imgind=randperm(length(desc_train));
    for i=1:nti
        d=desc_train(imgind(i));
        clf, showimage(imread(strrep(d.imgfname,'_train','')));
        x=d.c;
        y=d.r;
        rad=d.rad;
        showcirclefeaturesrad([x,y,rad]);
        title(sprintf('%d features in %s',length(d.c),d.imgfname));
        pause
    end
end


%% Load pre-computed SIFT features for test images 

lasti=1;
for i = 1:length(data)
     images_descs = get_descriptors_files(data,i,file_ext,desc_name,'test');
     for j = 1:length(images_descs) 
        fname = fullfile(basepath,'img',dataset_dir,data(i).classname,images_descs{j});
        fprintf('Loading %s /n',fname);
        tmp = load(fname,'-mat');
        tmp.desc.class=i;
        tmp.desc.imgfname=regexprep(fname,['.' desc_name],'.jpg');
        desc_test(lasti)=tmp.desc;
        desc_test(lasti).sift = single(desc_test(lasti).sift);
        lasti=lasti+1;
     end
end


%% Build visual vocabulary using k-means %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if do_form_codebook
    fprintf('/nBuild visual vocabulary:/n');

    % concatenate all descriptors from all images into a n x d matrix 
    DESC = [];
    labels_train = cat(1,desc_train.class);
    for i=1:length(data)
        desc_class = desc_train(labels_train==i);
        randimages = randperm(num_train_img);
        randimages=randimages(1:5);
        DESC = vertcat(DESC,desc_class(randimages).sift);
    end

    % sample random M (e.g. M=20,000) descriptors from all training descriptors
    r = randperm(size(DESC,1));
    r = r(1:min(length(r),nfeat_codebook));

    DESC = DESC(r,:);

    % run k-means
    K = nwords_codebook; % size of visual vocabulary
    fprintf('running k-means clustering of %d points into %d clusters.../n',...
        size(DESC,1),K)
    % input matrix needs to be transposed as the k-means function expects 
    % one point per column rather than per row

    % form options structure for clustering
    cluster_options.maxiters = max_km_iters;
    cluster_options.verbose  = 1;

    [VC] = kmeans_bo(double(DESC),K,max_km_iters);%visual codebook
    VC = VC';%transpose for compatibility with following functions
    clear DESC;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% Part 2: represent images with BOF histograms %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%   EXERCISE 2: Bag-of-Features image classification                      %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Represent each image by the normalized histogram of visual
% word labels of its features. Compute word histogram H over 
% the whole image, normalize histograms w.r.t. L1-norm.
%
% TODO:
% 2.1 for each training and test image compute H. Hint: use
%     Matlab function 'histc' to compute histograms.


N = size(VC,1); % number of visual words

for i=1:length(desc_train) 
    visword = desc_train(i).visword;    
    
    %H =...
  
    % normalize bow-hist (L1 norm)
    % ...
  
    % save histograms
    %desc_train(i).bof = ...
end

for i=1:length(desc_test) 
    visword = desc_test(i).visword;  
    
    %H =...
  
    % normalize bow-hist (L1 norm)
    % ...
    
    % save histograms
    %desc_test(i).bof = ...
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   End of EXERCISE 2                                                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%