
function provaTest(evaluate_model, image_to_test, basepath, do_feat_extraction_test,desc_name, VC ) 
dataset_dir_test = '/test_set/split_by_class_RGB';
clearvars data_test desc_test bof_l2lab;
% Create a new dataset split
if evaluate_model
    %test set
    data_test = create_dataset_split_structure(strcat(basepath, 'img/egocart'), 0, 100, 'jpg');
else 
    %single image
    data_test = create_single_image_split_structure(strcat(basepath, 'img/egocart'), image_to_test);
end

% Extract SIFT features from test images
if do_feat_extraction_test  
    if evaluate_model
        extract_sift_features(fullfile(basepath,'img/egocart','/test_set/split_by_class_RGB/'),desc_name);    
    else
        extract_sift_features(fullfile(basepath,'img/egocart',image_to_test),desc_name);    
    end
end
%% Load pre-computed SIFT features for test images 
lasti=1;
for i = 1:length(data_test)
     images_descs = get_descriptors_files(data_test,i,'jpg',desc_name,'test');
     for j = 1:length(images_descs)
        fname = fullfile(basepath,'img/egocart',dataset_dir_test,data_test(i).classname,images_descs{j});
        fprintf('Loading %s \n',fname);
        tmp = load(fname,'-mat');
        tmp.desc.class=i;
        tmp.desc.imgfname=regexprep(fname,['.' desc_name],'.jpg');
        desc_test(lasti)=tmp.desc;
        desc_test(lasti).sift = single(desc_test(lasti).sift);
        lasti=lasti+1;
    end;
end;
%% K-means descriptor quantization means assignment of each feature

 for i=1:length(desc_test)    
      sift = desc_test(i).sift(:,:); 
      dmat = eucliddist(sift,VC);
      [quantdist,visword] = min(dmat,[],2);
      % save feature labels
      desc_test(i).visword = visword;
      desc_test(i).quantdist = quantdist;
 end

for i=1:length(desc_test) 
    visword = desc_test(i).visword;
    H = histc(visword,[1:nwords_codebook]);
  
    % normalize bow-hist (L1 norm)
    if norm_bof_hist
        H = H/sum(H);
    end
  
    % save histograms
    desc_test(i).bof=H(:)';
end

for i=1:length(desc_test) 
   disp(desc_test(i).imgfname);
   desc_test(i).llc = max(LLC_coding_appr(VC,desc_test(i).sift));
   desc_test(i).llc=desc_test(i).llc/norm(desc_test(i).llc);
end

bof_test=cat(1,desc_test.bof);
llc_test = cat(1,desc_test.llc);
labels_test=cat(1,desc_test.class);


if evaluate_model
    %% NN classification %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % Compute L2 distance between BOFs of test and training images
        bof_l2dist=eucliddist(bof_test,bof_train);

        % Nearest neighbor classification (1-NN) using L2 distance
        [mv,mi] = min(bof_l2dist,[],2);
        bof_l2lab = labels_train(mi);

        method_name='NN L2';
        acc=sum(bof_l2lab==labels_test)/length(labels_test);
        fprintf('\n*** %s ***\nAccuracy = %1.4f%% (classification)\n',method_name,acc*100);

        % Compute classification accuracy
        compute_accuracy(data,labels_test,bof_l2lab,classes,method_name,desc_test,...
                          visualize_confmat & have_screen,... 
                          visualize_res & have_screen);
    



    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                                                                         %
    %   EXERCISE 3: Image classification                                      %
    %                                                                         %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %% Repeat Nearest Neighbor image classification using Chi2 distance
    % instead of L2. Hint: Chi2 distance between two row-vectors A,B can  
    % be computed with d=chi2(A,B);
    %
    % TODO:
    % 3.1 Nearest Neighbor classification with Chi2 distance
    %     Compute and compare overall and per-class classification
    %     accuracies to the L2 classification above


         % compute pair-wise CHI2
        bof_chi2dist = zeros(size(bof_test,1),size(bof_train,1));
        
        % bof_chi2dist = slmetric_pw(bof_train, bof_test, 'chisq');
        for i = 1:size(bof_test,1)
            for j = 1:size(bof_train,1)
                bof_chi2dist(i,j) = chi2(bof_test(i,:),bof_train(j,:)); 
            end
        end
    
        % Nearest neighbor classification (1-NN) using Chi2 distance
        [mv,mi] = min(bof_chi2dist,[],2);
        bof_chi2lab = labels_train(mi);
    
        method_name='NN Chi-2';
        acc=sum(bof_chi2lab==labels_test)/length(labels_test);
        fprintf('*** %s ***\nAccuracy = %1.4f%% (classification)\n',method_name,acc*100);
     
        % Compute classification accuracy
        compute_accuracy(data,labels_test,bof_chi2lab,classes,method_name,desc_test,...
                          visualize_confmat & have_screen,... 
                          visualize_res & have_screen);
    

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %   End of EXERCISE 3.1                                                   %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
else
    bof_l2dist=eucliddist(bof_test,bof_train);

    % Nearest neighbor classification (1-NN) using L2 distance
    [mv,mi] = min(bof_l2dist,[],2);
    bof_l2lab = labels_train(mi);
    fprintf("Predict %s, Correct %s", bof_l2lab, labels_test);
end