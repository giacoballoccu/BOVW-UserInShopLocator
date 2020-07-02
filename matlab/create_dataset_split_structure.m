%This function takes as input the directory containing the dataset.
%For example if we have 4 categories, say airplanes,faces,motorbikes and
%cars directory structure should be:   ./caltech4
%                                      ./caltech4/airplanes
%                                      ./caltech4/faces
%                                      ./caltech4/motorbikes
%                                      ./caltech4/cars
% This functions creates a random split of the dataset. For each category 
% selects Ntrain training images and min(N-Ntrain,Ntest) test images, where
% N is the amount of images of a given category.
% outputs a structure array with the following fields
%    n_images: 1074
%    classname: 'airplanes'; 
%    files: {1x1074 cell}; cell array with file names withouth path, e.g. img_100.jpg
%    train_id: [1x1074 logical]; Boolean array indicating training files
%    test_id: [1x1074 logical];  Boolean array indicating test files                                   
function data = create_dataset_split_structure(main_dir, file_ext)
    category_dirs_train_path = strcat(main_dir, '/train_set/split_by_class_RGB')
    category_dirs_train = dir(category_dirs_train_path);
    category_dirs_test_path = strcat(main_dir, '/test_set/split_by_class_RGB');
    category_dirs_test = dir(category_dirs_test_path);

    %remove '..' and '.' directories
    category_dirs_train(~cellfun(@isempty, regexp({category_dirs_train.name}, '\.*')))=[];
    category_dirs_train(strcmp({category_dirs_train.name},'split.mat'))=[]; 
    
    category_dirs_test(~cellfun(@isempty, regexp({category_dirs_test.name}, '\.*')))=[];
    category_dirs_test(strcmp({category_dirs_test.name},'split.mat'))=[]; 
    
    for c = 1:length(category_dirs_train)
        if isdir(fullfile(category_dirs_train_path,category_dirs_train(c).name)) && ~strcmp(category_dirs_train(c).name,'.') ...
                && ~strcmp(category_dirs_train(c).name,'..')
            imgdir = dir(fullfile(category_dirs_train_path,category_dirs_train(c).name, ['*.' file_ext]));
            data(c).n_images = length(imgdir);
            data(c).classname = category_dirs_train(c).name;
            data(c).files = {imgdir(:).name};
            data(c).train_id = true(1,data(c).n_images);
            data(c).test_id = false(1,data(c).n_images);

        end
    end
    
%       for c = 1:length(category_dirs_test)
%         if isdir(fullfile(category_dirs_test_path,category_dirs_test(c).name)) && ~strcmp(category_dirs_test(c).name,'.') ...
%                 && ~strcmp(category_dirs_test(c).name,'..')
%             imgdir = dir(fullfile(category_dirs_test_path,category_dirs_test(c).name, ['*.' file_ext]));
%             data(c).n_images = length(imgdir);
%             data(c).classname = category_dirs_test(c).name;
%             data(c).files = {imgdir(:).name};
%             data(c).train_id = false(1,data(c).n_images);
%             data(c).test_id = true(1,data(c).n_images);
%         end
%     end
end
