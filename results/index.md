# Fu-En Wang(102061149)
# Project3: Scene recognition with bag of words

## Introduction
In this project we will implement several system to recongnize what kind of scene in 1500 test images.

## Background Knowledge
### SIFT
Scale-invariant feature transform (SIFT) can detect the key point loacation and descriptor of an image. Because SIFT will calculate the orientation of patches, so it is commonly considered robust to view change of camera.

### Kmeans
When we has a set feature represetation and we want to divide then into K groups, we can simply use kmeans algorithm to do this.

### Support Vector Machine(SVM)
SVM is a common way to classify the category of data in machine learning. SVM will train the weight parameter according a binary label with 1 for true positive, -1 for true negative.

## Implementation

### Feature Representation
#### Tiny Image
<strong>Tiny Image</strong> is to resize the origin image to certain size(i.e 16 x 16) and horizontally vectorized it.
```
    image_feats = []; 
    for i = 1:length(image_paths)
        img = imread(image_paths{i});
        img_resize = imresize(img, [16, 16]);
        image_feats(end+1, :) = img_resize(:)';
    end 
```
#### Bag of SIFT
<strong>First</strong>, we need to extract the vocabulary in our train image sets. The method is to random sample SIFT features from each train images, for me I sample 256 features per image. And then we calculate kmeans of these features.
```
    sample_desc = []; 
    vocab = 0;
    sample_num = 1000;
    sample_image_index = randsample(1:length(image_paths), sample_num);
    for i = 1:length(sample_image_index)
        fprintf('Vocab:  %d', i)
        img = im2single(imread(image_paths{sample_image_index(i)}));
        [loc, feature] = vl_dsift(img, 'fast');
        %[loc, feature] = vl_sift(img);
        sample_index = randsample(1:size(loc,2), 256);
        sample_desc = [sample_desc feature(:, sample_index)];
    end 
    [vocab, assignments] = vl_kmeans(single(sample_desc), vocab_size);
```

<strong>Second</strong>, we will calculate the euclidean distance between feature in each test images and the vocabulary. For each test image, we will match each feature to a group which is the index of the shortest distance. As a result, we will get a histogram based on our vocabulary. And the histogram is the feature vector of an image.  
```
    load('vocab.mat');
    vocab_size = size(vocab, 2); 
    image_feats = zeros([length(image_paths) vocab_size]);
    for i = 1:length(image_paths)
        fprintf('bag:   %d', i)
        img = single(imread(image_paths{i}));
        [loc, desc] = vl_dsift(img, 'step', 5, 'fast');
        D = vl_alldist2(single(desc), vocab);
        [~, min_idx_map] = min(D, [], 2); 
        for j = 1:vocab_size
            image_feats(i, j) = sum(min_idx_map == j); 
        end
    end 
```

### Classifier
#### Nearest Neighbor
Nearest Neighbor is simply calculate the euclidean distance between train image and  test image features and choose the category which the closest one belong to.
```
    predicted_categories = {}; 
    D = vl_alldist2(test_image_feats', train_image_feats');
    [~, min_index_map] = min(D, [], 2); 
    size(min_index_map)
    max(min_index_map)
    for i=1:size(test_image_feats, 1)
        predicted_categories{end+1} = train_labels{min_index_map(i)};
    end 
    predicted_categories = predicted_categories'
```
#### Support Vector Machine(SVM)
We can use SVM to train the weight for each category to obtain the 1 vs all model. And we can simply use this model to weight test image feature and the category which the one has highest score belong to.
```
    categories = unique(train_labels); 
    num_categories = length(categories);
    train_feat = train_image_feats';
    test_feat = test_image_feats';
    W_mat = [];
    B_lst = [];
    %%%%
    LAMBDA = 0.01;
    diag_freq = 100;
    maxIter = 1E5;
    SVMeps = 1E-5;
    %%%%
    for i = 1:length(categories)
        categories{i}
        indice = strcmp(categories{i}, train_labels);
        binary = zeros([1 length(train_labels)]) - 1;
        binary(indice) = 1;
        %binary
        %error('GG')
        %binary(~binary) = -1;
        [W B] = vl_svmtrain( train_feat, binary, LAMBDA, 'MaxNumIterations', maxIter, ...
                            'Epsilon',SVMeps);
        W_mat(end+1, :) = W;
        B_lst(end+1) = B;
    end
    B_mat = [];
    for i=1:size(test_feat,2)
        B_mat = [B_mat B_lst'];
    end
    score = W_mat * test_feat + B_mat;
    [~, max_index] = max(score, [], 1);
    predicted_categories = {};
    for i = 1:size(test_feat,2)
        predicted_categories{end+1} = categories{max_index(i)};
    end
    predicted_categories = predicted_categories';
```
## Result
### Tiny Image and Nearest Neighbor
<center>
<img src="confusion_matrix1.png">

<br>
Accuracy (mean of diagonal of confusion matrix) is 0.191
<p>

<table border=0 cellpadding=4 cellspacing=1>
<tr>
<th>Category name</th>
<th>Accuracy</th>
<th colspan=2>Sample training images</th>
<th colspan=2>Sample true positives</th>
<th colspan=2>False positives with true label</th>
<th colspan=2>False negatives with wrong predicted label</th>
</tr>
<tr>
<td>Kitchen</td>
<td>0.050</td>
<td bgcolor=LightBlue><img src="thumbnails/Kitchen_image_0172.jpg" width=57 height=75></td>
<td bgcolor=LightBlue><img src="thumbnails/Kitchen_image_0136.jpg" width=101 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Kitchen_image_0107.jpg" width=101 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Kitchen_image_0190.jpg" width=57 height=75></td>
<td bgcolor=LightCoral><img src="thumbnails/InsideCity_image_0002.jpg" width=75 height=75><br><small>InsideCity</small></td>
<td bgcolor=LightCoral><img src="thumbnails/Industrial_image_0105.jpg" width=113 height=75><br><small>Industrial</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Kitchen_image_0190.jpg" width=57 height=75><br><small>Kitchen</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Kitchen_image_0110.jpg" width=100 height=75><br><small>Kitchen</small></td>
</tr>
<tr>
<td>Store</td>
<td>0.020</td>
<td bgcolor=LightBlue><img src="thumbnails/Store_image_0007.jpg" width=100 height=75></td>
<td bgcolor=LightBlue><img src="thumbnails/Store_image_0298.jpg" width=100 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Store_image_0057.jpg" width=100 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Store_image_0009.jpg" width=100 height=75></td>
<td bgcolor=LightCoral><img src="thumbnails/Bedroom_image_0020.jpg" width=101 height=75><br><small>Bedroom</small></td>
<td bgcolor=LightCoral><img src="thumbnails/Bedroom_image_0014.jpg" width=101 height=75><br><small>Bedroom</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Store_image_0009.jpg" width=100 height=75><br><small>Store</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Store_image_0057.jpg" width=100 height=75><br><small>Store</small></td>
</tr>
<tr>
<td>Bedroom</td>
<td>0.080</td>
<td bgcolor=LightBlue><img src="thumbnails/Bedroom_image_0045.jpg" width=57 height=75></td>
<td bgcolor=LightBlue><img src="thumbnails/Bedroom_image_0154.jpg" width=84 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Bedroom_image_0090.jpg" width=108 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Bedroom_image_0180.jpg" width=78 height=75></td>
<td bgcolor=LightCoral><img src="thumbnails/Mountain_image_0025.jpg" width=75 height=75><br><small>Mountain</small></td>
<td bgcolor=LightCoral><img src="thumbnails/InsideCity_image_0069.jpg" width=75 height=75><br><small>InsideCity</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Bedroom_image_0034.jpg" width=101 height=75><br><small>Bedroom</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Bedroom_image_0119.jpg" width=113 height=75><br><small>Bedroom</small></td>
</tr>
<tr>
<td>LivingRoom</td>
<td>0.080</td>
<td bgcolor=LightBlue><img src="thumbnails/LivingRoom_image_0231.jpg" width=85 height=75></td>
<td bgcolor=LightBlue><img src="thumbnails/LivingRoom_image_0103.jpg" width=100 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/LivingRoom_image_0005.jpg" width=93 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/LivingRoom_image_0021.jpg" width=109 height=75></td>
<td bgcolor=LightCoral><img src="thumbnails/Office_image_0176.jpg" width=109 height=75><br><small>Office</small></td>
<td bgcolor=LightCoral><img src="thumbnails/Bedroom_image_0056.jpg" width=113 height=75><br><small>Bedroom</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/LivingRoom_image_0021.jpg" width=109 height=75><br><small>LivingRoom</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/LivingRoom_image_0003.jpg" width=114 height=75><br><small>LivingRoom</small></td>
</tr>
<tr>
<td>Office</td>
<td>0.050</td>
<td bgcolor=LightBlue><img src="thumbnails/Office_image_0014.jpg" width=83 height=75></td>
<td bgcolor=LightBlue><img src="thumbnails/Office_image_0178.jpg" width=122 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Office_image_0023.jpg" width=90 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Office_image_0114.jpg" width=117 height=75></td>
<td bgcolor=LightCoral><img src="thumbnails/Coast_image_0031.jpg" width=75 height=75><br><small>Coast</small></td>
<td bgcolor=LightCoral><img src="thumbnails/Kitchen_image_0097.jpg" width=101 height=75><br><small>Kitchen</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Office_image_0144.jpg" width=115 height=75><br><small>Office</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Office_image_0022.jpg" width=102 height=75><br><small>Office</small></td>
</tr>
<tr>
<td>Industrial</td>
<td>0.020</td>
<td bgcolor=LightBlue><img src="thumbnails/Industrial_image_0143.jpg" width=100 height=75></td>
<td bgcolor=LightBlue><img src="thumbnails/Industrial_image_0173.jpg" width=95 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Industrial_image_0107.jpg" width=50 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Industrial_image_0038.jpg" width=94 height=75></td>
<td bgcolor=LightCoral><img src="thumbnails/Mountain_image_0047.jpg" width=75 height=75><br><small>Mountain</small></td>
<td bgcolor=LightCoral><img src="thumbnails/Street_image_0017.jpg" width=75 height=75><br><small>Street</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Industrial_image_0038.jpg" width=94 height=75><br><small>Industrial</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Industrial_image_0107.jpg" width=50 height=75><br><small>Industrial</small></td>
</tr>
<tr>
<td>Suburb</td>
<td>0.210</td>
<td bgcolor=LightBlue><img src="thumbnails/Suburb_image_0235.jpg" width=113 height=75></td>
<td bgcolor=LightBlue><img src="thumbnails/Suburb_image_0109.jpg" width=113 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Suburb_image_0103.jpg" width=113 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Suburb_image_0018.jpg" width=113 height=75></td>
<td bgcolor=LightCoral><img src="thumbnails/Store_image_0140.jpg" width=98 height=75><br><small>Store</small></td>
<td bgcolor=LightCoral></td>
<td bgcolor=#FFBB55><img src="thumbnails/Suburb_image_0023.jpg" width=113 height=75><br><small>Suburb</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Suburb_image_0116.jpg" width=113 height=75><br><small>Suburb</small></td>
</tr>
<tr>
<td>InsideCity</td>
<td>0.100</td>
<td bgcolor=LightBlue><img src="thumbnails/InsideCity_image_0005.jpg" width=75 height=75></td>
<td bgcolor=LightBlue><img src="thumbnails/InsideCity_image_0047.jpg" width=75 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/InsideCity_image_0094.jpg" width=75 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/InsideCity_image_0130.jpg" width=75 height=75></td>
<td bgcolor=LightCoral><img src="thumbnails/Mountain_image_0119.jpg" width=75 height=75><br><small>Mountain</small></td>
<td bgcolor=LightCoral><img src="thumbnails/OpenCountry_image_0113.jpg" width=75 height=75><br><small>OpenCountry</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/InsideCity_image_0024.jpg" width=75 height=75><br><small>InsideCity</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/InsideCity_image_0130.jpg" width=75 height=75><br><small>InsideCity</small></td>
</tr>
<tr>
<td>TallBuilding</td>
<td>0.110</td>
<td bgcolor=LightBlue><img src="thumbnails/TallBuilding_image_0307.jpg" width=75 height=75></td>
<td bgcolor=LightBlue><img src="thumbnails/TallBuilding_image_0071.jpg" width=75 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/TallBuilding_image_0005.jpg" width=75 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/TallBuilding_image_0037.jpg" width=75 height=75></td>
<td bgcolor=LightCoral><img src="thumbnails/InsideCity_image_0068.jpg" width=75 height=75><br><small>InsideCity</small></td>
<td bgcolor=LightCoral><img src="thumbnails/Office_image_0148.jpg" width=100 height=75><br><small>Office</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/TallBuilding_image_0037.jpg" width=75 height=75><br><small>TallBuilding</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/TallBuilding_image_0054.jpg" width=75 height=75><br><small>TallBuilding</small></td>
</tr>
<tr>
<td>Street</td>
<td>0.400</td>
<td bgcolor=LightBlue><img src="thumbnails/Street_image_0004.jpg" width=75 height=75></td>
<td bgcolor=LightBlue><img src="thumbnails/Street_image_0236.jpg" width=75 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Street_image_0129.jpg" width=75 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Street_image_0122.jpg" width=75 height=75></td>
<td bgcolor=LightCoral><img src="thumbnails/Suburb_image_0161.jpg" width=113 height=75><br><small>Suburb</small></td>
<td bgcolor=LightCoral><img src="thumbnails/Suburb_image_0149.jpg" width=113 height=75><br><small>Suburb</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Street_image_0014.jpg" width=75 height=75><br><small>Street</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Street_image_0066.jpg" width=75 height=75><br><small>Street</small></td>
</tr>
<tr>
<td>Highway</td>
<td>0.690</td>
<td bgcolor=LightBlue><img src="thumbnails/Highway_image_0054.jpg" width=75 height=75></td>
<td bgcolor=LightBlue><img src="thumbnails/Highway_image_0038.jpg" width=75 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Highway_image_0004.jpg" width=75 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Highway_image_0137.jpg" width=75 height=75></td>
<td bgcolor=LightCoral><img src="thumbnails/Suburb_image_0162.jpg" width=113 height=75><br><small>Suburb</small></td>
<td bgcolor=LightCoral><img src="thumbnails/Forest_image_0105.jpg" width=75 height=75><br><small>Forest</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Highway_image_0100.jpg" width=75 height=75><br><small>Highway</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Highway_image_0027.jpg" width=75 height=75><br><small>Highway</small></td>
</tr>
<tr>
<td>OpenCountry</td>
<td>0.310</td>
<td bgcolor=LightBlue><img src="thumbnails/OpenCountry_image_0141.jpg" width=75 height=75></td>
<td bgcolor=LightBlue><img src="thumbnails/OpenCountry_image_0308.jpg" width=75 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/OpenCountry_image_0106.jpg" width=75 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/OpenCountry_image_0082.jpg" width=75 height=75></td>
<td bgcolor=LightCoral><img src="thumbnails/Office_image_0134.jpg" width=100 height=75><br><small>Office</small></td>
<td bgcolor=LightCoral><img src="thumbnails/Store_image_0010.jpg" width=104 height=75><br><small>Store</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/OpenCountry_image_0091.jpg" width=75 height=75><br><small>OpenCountry</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/OpenCountry_image_0125.jpg" width=75 height=75><br><small>OpenCountry</small></td>
</tr>
<tr>
<td>Coast</td>
<td>0.280</td>
<td bgcolor=LightBlue><img src="thumbnails/Coast_image_0034.jpg" width=75 height=75></td>
<td bgcolor=LightBlue><img src="thumbnails/Coast_image_0200.jpg" width=75 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Coast_image_0046.jpg" width=75 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Coast_image_0003.jpg" width=75 height=75></td>
<td bgcolor=LightCoral><img src="thumbnails/Highway_image_0017.jpg" width=75 height=75><br><small>Highway</small></td>
<td bgcolor=LightCoral><img src="thumbnails/Bedroom_image_0071.jpg" width=112 height=75><br><small>Bedroom</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Coast_image_0041.jpg" width=75 height=75><br><small>Coast</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Coast_image_0018.jpg" width=75 height=75><br><small>Coast</small></td>
</tr>
<tr>
<td>Mountain</td>
<td>0.130</td>
<td bgcolor=LightBlue><img src="thumbnails/Mountain_image_0276.jpg" width=75 height=75></td>
<td bgcolor=LightBlue><img src="thumbnails/Mountain_image_0324.jpg" width=75 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Mountain_image_0062.jpg" width=75 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Mountain_image_0088.jpg" width=75 height=75></td>
<td bgcolor=LightCoral><img src="thumbnails/Suburb_image_0061.jpg" width=113 height=75><br><small>Suburb</small></td>
<td bgcolor=LightCoral><img src="thumbnails/OpenCountry_image_0052.jpg" width=75 height=75><br><small>OpenCountry</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Mountain_image_0062.jpg" width=75 height=75><br><small>Mountain</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Mountain_image_0088.jpg" width=75 height=75><br><small>Mountain</small></td>
</tr>
<tr>
<td>Forest</td>
<td>0.340</td>
<td bgcolor=LightBlue><img src="thumbnails/Forest_image_0192.jpg" width=75 height=75></td>
<td bgcolor=LightBlue><img src="thumbnails/Forest_image_0140.jpg" width=75 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Forest_image_0027.jpg" width=75 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Forest_image_0043.jpg" width=75 height=75></td>
<td bgcolor=LightCoral><img src="thumbnails/InsideCity_image_0016.jpg" width=75 height=75><br><small>InsideCity</small></td>
<td bgcolor=LightCoral><img src="thumbnails/Highway_image_0026.jpg" width=75 height=75><br><small>Highway</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Forest_image_0080.jpg" width=75 height=75><br><small>Forest</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Forest_image_0064.jpg" width=75 height=75><br><small>Forest</small></td>
</tr>
<tr>
<th>Category name</th>
<th>Accuracy</th>
<th colspan=2>Sample training images</th>
<th colspan=2>Sample true positives</th>
<th colspan=2>False positives with true label</th>
<th colspan=2>False negatives with wrong predicted label</th>
</tr>
</table>
</center>

### Bag of SIFT and Nearest Neighbor
<center>
<img src="confusion_matrix2.png">

<br>
Accuracy (mean of diagonal of confusion matrix) is 0.523
<p>

<table border=0 cellpadding=4 cellspacing=1>
<tr>
<th>Category name</th>
<th>Accuracy</th>
<th colspan=2>Sample training images</th>
<th colspan=2>Sample true positives</th>
<th colspan=2>False positives with true label</th>
<th colspan=2>False negatives with wrong predicted label</th>
</tr>
<tr>
<td>Kitchen</td>
<td>0.430</td>
<td bgcolor=LightBlue><img src="thumbnails/Kitchen_image_0131.jpg" width=100 height=75></td>
<td bgcolor=LightBlue><img src="thumbnails/Kitchen_image_0004.jpg" width=100 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Kitchen_image_0097.jpg" width=101 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Kitchen_image_0002.jpg" width=100 height=75></td>
<td bgcolor=LightCoral><img src="thumbnails/Bedroom_image_0054.jpg" width=100 height=75><br><small>Bedroom</small></td>
<td bgcolor=LightCoral><img src="thumbnails/Bedroom_image_0166.jpg" width=101 height=75><br><small>Bedroom</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Kitchen_image_0097.jpg" width=101 height=75><br><small>Kitchen</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Kitchen_image_0006.jpg" width=100 height=75><br><small>Kitchen</small></td>
</tr>
<tr>
<td>Store</td>
<td>0.550</td>
<td bgcolor=LightBlue><img src="thumbnails/Store_image_0035.jpg" width=113 height=75></td>
<td bgcolor=LightBlue><img src="thumbnails/Store_image_0232.jpg" width=113 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Store_image_0104.jpg" width=100 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Store_image_0073.jpg" width=101 height=75></td>
<td bgcolor=LightCoral><img src="thumbnails/Coast_image_0047.jpg" width=75 height=75><br><small>Coast</small></td>
<td bgcolor=LightCoral><img src="thumbnails/Industrial_image_0118.jpg" width=116 height=75><br><small>Industrial</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Store_image_0114.jpg" width=100 height=75><br><small>Store</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Store_image_0053.jpg" width=57 height=75><br><small>Store</small></td>
</tr>
<tr>
<td>Bedroom</td>
<td>0.410</td>
<td bgcolor=LightBlue><img src="thumbnails/Bedroom_image_0169.jpg" width=115 height=75></td>
<td bgcolor=LightBlue><img src="thumbnails/Bedroom_image_0133.jpg" width=113 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Bedroom_image_0029.jpg" width=133 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Bedroom_image_0003.jpg" width=104 height=75></td>
<td bgcolor=LightCoral><img src="thumbnails/Highway_image_0017.jpg" width=75 height=75><br><small>Highway</small></td>
<td bgcolor=LightCoral><img src="thumbnails/LivingRoom_image_0087.jpg" width=100 height=75><br><small>LivingRoom</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Bedroom_image_0104.jpg" width=95 height=75><br><small>Bedroom</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Bedroom_image_0074.jpg" width=116 height=75><br><small>Bedroom</small></td>
</tr>
<tr>
<td>LivingRoom</td>
<td>0.340</td>
<td bgcolor=LightBlue><img src="thumbnails/LivingRoom_image_0208.jpg" width=70 height=75></td>
<td bgcolor=LightBlue><img src="thumbnails/LivingRoom_image_0261.jpg" width=100 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/LivingRoom_image_0066.jpg" width=101 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/LivingRoom_image_0095.jpg" width=100 height=75></td>
<td bgcolor=LightCoral><img src="thumbnails/Kitchen_image_0141.jpg" width=51 height=75><br><small>Kitchen</small></td>
<td bgcolor=LightCoral><img src="thumbnails/TallBuilding_image_0129.jpg" width=75 height=75><br><small>TallBuilding</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/LivingRoom_image_0106.jpg" width=101 height=75><br><small>LivingRoom</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/LivingRoom_image_0066.jpg" width=101 height=75><br><small>LivingRoom</small></td>
</tr>
<tr>
<td>Office</td>
<td>0.660</td>
<td bgcolor=LightBlue><img src="thumbnails/Office_image_0139.jpg" width=117 height=75></td>
<td bgcolor=LightBlue><img src="thumbnails/Office_image_0073.jpg" width=117 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Office_image_0040.jpg" width=103 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Office_image_0032.jpg" width=120 height=75></td>
<td bgcolor=LightCoral><img src="thumbnails/Suburb_image_0102.jpg" width=113 height=75><br><small>Suburb</small></td>
<td bgcolor=LightCoral><img src="thumbnails/Kitchen_image_0167.jpg" width=57 height=75><br><small>Kitchen</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Office_image_0075.jpg" width=103 height=75><br><small>Office</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Office_image_0169.jpg" width=92 height=75><br><small>Office</small></td>
</tr>
<tr>
<td>Industrial</td>
<td>0.300</td>
<td bgcolor=LightBlue><img src="thumbnails/Industrial_image_0136.jpg" width=100 height=75></td>
<td bgcolor=LightBlue><img src="thumbnails/Industrial_image_0070.jpg" width=118 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Industrial_image_0108.jpg" width=100 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Industrial_image_0066.jpg" width=123 height=75></td>
<td bgcolor=LightCoral><img src="thumbnails/TallBuilding_image_0106.jpg" width=75 height=75><br><small>TallBuilding</small></td>
<td bgcolor=LightCoral><img src="thumbnails/Street_image_0079.jpg" width=75 height=75><br><small>Street</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Industrial_image_0015.jpg" width=55 height=75><br><small>Industrial</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Industrial_image_0060.jpg" width=111 height=75><br><small>Industrial</small></td>
</tr>
<tr>
<td>Suburb</td>
<td>0.770</td>
<td bgcolor=LightBlue><img src="thumbnails/Suburb_image_0169.jpg" width=113 height=75></td>
<td bgcolor=LightBlue><img src="thumbnails/Suburb_image_0159.jpg" width=113 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Suburb_image_0062.jpg" width=113 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Suburb_image_0118.jpg" width=113 height=75></td>
<td bgcolor=LightCoral><img src="thumbnails/Forest_image_0036.jpg" width=75 height=75><br><small>Forest</small></td>
<td bgcolor=LightCoral><img src="thumbnails/Mountain_image_0091.jpg" width=75 height=75><br><small>Mountain</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Suburb_image_0004.jpg" width=113 height=75><br><small>Suburb</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Suburb_image_0161.jpg" width=113 height=75><br><small>Suburb</small></td>
</tr>
<tr>
<td>InsideCity</td>
<td>0.380</td>
<td bgcolor=LightBlue><img src="thumbnails/InsideCity_image_0028.jpg" width=75 height=75></td>
<td bgcolor=LightBlue><img src="thumbnails/InsideCity_image_0070.jpg" width=75 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/InsideCity_image_0079.jpg" width=75 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/InsideCity_image_0022.jpg" width=75 height=75></td>
<td bgcolor=LightCoral><img src="thumbnails/Street_image_0100.jpg" width=75 height=75><br><small>Street</small></td>
<td bgcolor=LightCoral><img src="thumbnails/TallBuilding_image_0053.jpg" width=75 height=75><br><small>TallBuilding</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/InsideCity_image_0094.jpg" width=75 height=75><br><small>InsideCity</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/InsideCity_image_0044.jpg" width=75 height=75><br><small>InsideCity</small></td>
</tr>
<tr>
<td>TallBuilding</td>
<td>0.300</td>
<td bgcolor=LightBlue><img src="thumbnails/TallBuilding_image_0033.jpg" width=75 height=75></td>
<td bgcolor=LightBlue><img src="thumbnails/TallBuilding_image_0280.jpg" width=75 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/TallBuilding_image_0019.jpg" width=75 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/TallBuilding_image_0088.jpg" width=75 height=75></td>
<td bgcolor=LightCoral><img src="thumbnails/InsideCity_image_0052.jpg" width=75 height=75><br><small>InsideCity</small></td>
<td bgcolor=LightCoral><img src="thumbnails/Mountain_image_0066.jpg" width=75 height=75><br><small>Mountain</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/TallBuilding_image_0114.jpg" width=75 height=75><br><small>TallBuilding</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/TallBuilding_image_0085.jpg" width=75 height=75><br><small>TallBuilding</small></td>
</tr>
<tr>
<td>Street</td>
<td>0.560</td>
<td bgcolor=LightBlue><img src="thumbnails/Street_image_0290.jpg" width=75 height=75></td>
<td bgcolor=LightBlue><img src="thumbnails/Street_image_0253.jpg" width=75 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Street_image_0014.jpg" width=75 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Street_image_0102.jpg" width=75 height=75></td>
<td bgcolor=LightCoral><img src="thumbnails/TallBuilding_image_0127.jpg" width=75 height=75><br><small>TallBuilding</small></td>
<td bgcolor=LightCoral><img src="thumbnails/InsideCity_image_0057.jpg" width=75 height=75><br><small>InsideCity</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Street_image_0074.jpg" width=75 height=75><br><small>Street</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Street_image_0131.jpg" width=75 height=75><br><small>Street</small></td>
</tr>
<tr>
<td>Highway</td>
<td>0.790</td>
<td bgcolor=LightBlue><img src="thumbnails/Highway_image_0012.jpg" width=75 height=75></td>
<td bgcolor=LightBlue><img src="thumbnails/Highway_image_0187.jpg" width=75 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Highway_image_0085.jpg" width=75 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Highway_image_0068.jpg" width=75 height=75></td>
<td bgcolor=LightCoral><img src="thumbnails/Mountain_image_0085.jpg" width=75 height=75><br><small>Mountain</small></td>
<td bgcolor=LightCoral><img src="thumbnails/Industrial_image_0072.jpg" width=122 height=75><br><small>Industrial</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Highway_image_0075.jpg" width=75 height=75><br><small>Highway</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Highway_image_0104.jpg" width=75 height=75><br><small>Highway</small></td>
</tr>
<tr>
<td>OpenCountry</td>
<td>0.470</td>
<td bgcolor=LightBlue><img src="thumbnails/OpenCountry_image_0282.jpg" width=75 height=75></td>
<td bgcolor=LightBlue><img src="thumbnails/OpenCountry_image_0074.jpg" width=75 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/OpenCountry_image_0086.jpg" width=75 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/OpenCountry_image_0038.jpg" width=75 height=75></td>
<td bgcolor=LightCoral><img src="thumbnails/Mountain_image_0118.jpg" width=75 height=75><br><small>Mountain</small></td>
<td bgcolor=LightCoral><img src="thumbnails/Coast_image_0071.jpg" width=75 height=75><br><small>Coast</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/OpenCountry_image_0086.jpg" width=75 height=75><br><small>OpenCountry</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/OpenCountry_image_0102.jpg" width=75 height=75><br><small>OpenCountry</small></td>
</tr>
<tr>
<td>Coast</td>
<td>0.420</td>
<td bgcolor=LightBlue><img src="thumbnails/Coast_image_0209.jpg" width=75 height=75></td>
<td bgcolor=LightBlue><img src="thumbnails/Coast_image_0287.jpg" width=75 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Coast_image_0076.jpg" width=75 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Coast_image_0072.jpg" width=75 height=75></td>
<td bgcolor=LightCoral><img src="thumbnails/Mountain_image_0060.jpg" width=75 height=75><br><small>Mountain</small></td>
<td bgcolor=LightCoral><img src="thumbnails/OpenCountry_image_0011.jpg" width=75 height=75><br><small>OpenCountry</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Coast_image_0040.jpg" width=75 height=75><br><small>Coast</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Coast_image_0017.jpg" width=75 height=75><br><small>Coast</small></td>
</tr>
<tr>
<td>Mountain</td>
<td>0.560</td>
<td bgcolor=LightBlue><img src="thumbnails/Mountain_image_0169.jpg" width=75 height=75></td>
<td bgcolor=LightBlue><img src="thumbnails/Mountain_image_0160.jpg" width=75 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Mountain_image_0042.jpg" width=75 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Mountain_image_0012.jpg" width=75 height=75></td>
<td bgcolor=LightCoral><img src="thumbnails/OpenCountry_image_0014.jpg" width=75 height=75><br><small>OpenCountry</small></td>
<td bgcolor=LightCoral><img src="thumbnails/Coast_image_0122.jpg" width=75 height=75><br><small>Coast</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Mountain_image_0041.jpg" width=75 height=75><br><small>Mountain</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Mountain_image_0014.jpg" width=75 height=75><br><small>Mountain</small></td>
</tr>
<tr>
<td>Forest</td>
<td>0.910</td>
<td bgcolor=LightBlue><img src="thumbnails/Forest_image_0092.jpg" width=75 height=75></td>
<td bgcolor=LightBlue><img src="thumbnails/Forest_image_0240.jpg" width=75 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Forest_image_0013.jpg" width=75 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Forest_image_0068.jpg" width=75 height=75></td>
<td bgcolor=LightCoral><img src="thumbnails/Industrial_image_0049.jpg" width=100 height=75><br><small>Industrial</small></td>
<td bgcolor=LightCoral><img src="thumbnails/Mountain_image_0079.jpg" width=75 height=75><br><small>Mountain</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Forest_image_0118.jpg" width=75 height=75><br><small>Forest</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Forest_image_0141.jpg" width=75 height=75><br><small>Forest</small></td>
</tr>
<tr>
<th>Category name</th>
<th>Accuracy</th>
<th colspan=2>Sample training images</th>
<th colspan=2>Sample true positives</th>
<th colspan=2>False positives with true label</th>
<th colspan=2>False negatives with wrong predicted label</th>
</tr>
</table>
</center>

### Bag of SIFT and SVM
<center>
<img src="confusion_matrix3.png">

<br>
Accuracy (mean of diagonal of confusion matrix) is 0.693
<p>

<table border=0 cellpadding=4 cellspacing=1>
<tr>
<th>Category name</th>
<th>Accuracy</th>
<th colspan=2>Sample training images</th>
<th colspan=2>Sample true positives</th>
<th colspan=2>False positives with true label</th>
<th colspan=2>False negatives with wrong predicted label</th>
</tr>
<tr>
<td>Kitchen</td>
<td>0.650</td>
<td bgcolor=LightBlue><img src="thumbnails/Kitchen_image_0161.jpg" width=100 height=75></td>
<td bgcolor=LightBlue><img src="thumbnails/Kitchen_image_0155.jpg" width=57 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Kitchen_image_0043.jpg" width=57 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Kitchen_image_0009.jpg" width=100 height=75></td>
<td bgcolor=LightCoral><img src="thumbnails/Office_image_0119.jpg" width=108 height=75><br><small>Office</small></td>
<td bgcolor=LightCoral><img src="thumbnails/LivingRoom_image_0078.jpg" width=113 height=75><br><small>LivingRoom</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Kitchen_image_0009.jpg" width=100 height=75><br><small>Kitchen</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Kitchen_image_0002.jpg" width=100 height=75><br><small>Kitchen</small></td>
</tr>
<tr>
<td>Store</td>
<td>0.600</td>
<td bgcolor=LightBlue><img src="thumbnails/Store_image_0103.jpg" width=106 height=75></td>
<td bgcolor=LightBlue><img src="thumbnails/Store_image_0092.jpg" width=100 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Store_image_0036.jpg" width=100 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Store_image_0052.jpg" width=100 height=75></td>
<td bgcolor=LightCoral><img src="thumbnails/InsideCity_image_0124.jpg" width=75 height=75><br><small>InsideCity</small></td>
<td bgcolor=LightCoral><img src="thumbnails/Street_image_0047.jpg" width=75 height=75><br><small>Street</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Store_image_0010.jpg" width=104 height=75><br><small>Store</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Store_image_0013.jpg" width=131 height=75><br><small>Store</small></td>
</tr>
<tr>
<td>Bedroom</td>
<td>0.360</td>
<td bgcolor=LightBlue><img src="thumbnails/Bedroom_image_0212.jpg" width=101 height=75></td>
<td bgcolor=LightBlue><img src="thumbnails/Bedroom_image_0155.jpg" width=113 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Bedroom_image_0087.jpg" width=50 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Bedroom_image_0077.jpg" width=90 height=75></td>
<td bgcolor=LightCoral><img src="thumbnails/Kitchen_image_0040.jpg" width=57 height=75><br><small>Kitchen</small></td>
<td bgcolor=LightCoral><img src="thumbnails/LivingRoom_image_0094.jpg" width=64 height=75><br><small>LivingRoom</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Bedroom_image_0014.jpg" width=101 height=75><br><small>Bedroom</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Bedroom_image_0063.jpg" width=115 height=75><br><small>Bedroom</small></td>
</tr>
<tr>
<td>LivingRoom</td>
<td>0.320</td>
<td bgcolor=LightBlue><img src="thumbnails/LivingRoom_image_0130.jpg" width=101 height=75></td>
<td bgcolor=LightBlue><img src="thumbnails/LivingRoom_image_0269.jpg" width=100 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/LivingRoom_image_0082.jpg" width=100 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/LivingRoom_image_0146.jpg" width=114 height=75></td>
<td bgcolor=LightCoral><img src="thumbnails/Street_image_0013.jpg" width=75 height=75><br><small>Street</small></td>
<td bgcolor=LightCoral><img src="thumbnails/Bedroom_image_0168.jpg" width=113 height=75><br><small>Bedroom</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/LivingRoom_image_0077.jpg" width=113 height=75><br><small>LivingRoom</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/LivingRoom_image_0095.jpg" width=100 height=75><br><small>LivingRoom</small></td>
</tr>
<tr>
<td>Office</td>
<td>0.980</td>
<td bgcolor=LightBlue><img src="thumbnails/Office_image_0141.jpg" width=102 height=75></td>
<td bgcolor=LightBlue><img src="thumbnails/Office_image_0078.jpg" width=116 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Office_image_0089.jpg" width=92 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Office_image_0083.jpg" width=108 height=75></td>
<td bgcolor=LightCoral><img src="thumbnails/InsideCity_image_0002.jpg" width=75 height=75><br><small>InsideCity</small></td>
<td bgcolor=LightCoral><img src="thumbnails/LivingRoom_image_0056.jpg" width=112 height=75><br><small>LivingRoom</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Office_image_0055.jpg" width=108 height=75><br><small>Office</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Office_image_0183.jpg" width=109 height=75><br><small>Office</small></td>
</tr>
<tr>
<td>Industrial</td>
<td>0.540</td>
<td bgcolor=LightBlue><img src="thumbnails/Industrial_image_0010.jpg" width=117 height=75></td>
<td bgcolor=LightBlue><img src="thumbnails/Industrial_image_0186.jpg" width=118 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Industrial_image_0104.jpg" width=112 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Industrial_image_0005.jpg" width=114 height=75></td>
<td bgcolor=LightCoral><img src="thumbnails/Kitchen_image_0177.jpg" width=100 height=75><br><small>Kitchen</small></td>
<td bgcolor=LightCoral><img src="thumbnails/TallBuilding_image_0004.jpg" width=75 height=75><br><small>TallBuilding</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Industrial_image_0098.jpg" width=100 height=75><br><small>Industrial</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Industrial_image_0045.jpg" width=61 height=75><br><small>Industrial</small></td>
</tr>
<tr>
<td>Suburb</td>
<td>0.980</td>
<td bgcolor=LightBlue><img src="thumbnails/Suburb_image_0131.jpg" width=113 height=75></td>
<td bgcolor=LightBlue><img src="thumbnails/Suburb_image_0096.jpg" width=113 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Suburb_image_0001.jpg" width=113 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Suburb_image_0005.jpg" width=113 height=75></td>
<td bgcolor=LightCoral><img src="thumbnails/Industrial_image_0026.jpg" width=97 height=75><br><small>Industrial</small></td>
<td bgcolor=LightCoral><img src="thumbnails/LivingRoom_image_0113.jpg" width=100 height=75><br><small>LivingRoom</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Suburb_image_0059.jpg" width=113 height=75><br><small>Suburb</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Suburb_image_0123.jpg" width=113 height=75><br><small>Suburb</small></td>
</tr>
<tr>
<td>InsideCity</td>
<td>0.640</td>
<td bgcolor=LightBlue><img src="thumbnails/InsideCity_image_0080.jpg" width=75 height=75></td>
<td bgcolor=LightBlue><img src="thumbnails/InsideCity_image_0199.jpg" width=75 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/InsideCity_image_0134.jpg" width=75 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/InsideCity_image_0109.jpg" width=75 height=75></td>
<td bgcolor=LightCoral><img src="thumbnails/Industrial_image_0079.jpg" width=101 height=75><br><small>Industrial</small></td>
<td bgcolor=LightCoral><img src="thumbnails/Industrial_image_0140.jpg" width=100 height=75><br><small>Industrial</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/InsideCity_image_0075.jpg" width=75 height=75><br><small>InsideCity</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/InsideCity_image_0062.jpg" width=75 height=75><br><small>InsideCity</small></td>
</tr>
<tr>
<td>TallBuilding</td>
<td>0.770</td>
<td bgcolor=LightBlue><img src="thumbnails/TallBuilding_image_0090.jpg" width=75 height=75></td>
<td bgcolor=LightBlue><img src="thumbnails/TallBuilding_image_0076.jpg" width=75 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/TallBuilding_image_0128.jpg" width=75 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/TallBuilding_image_0037.jpg" width=75 height=75></td>
<td bgcolor=LightCoral><img src="thumbnails/Store_image_0015.jpg" width=100 height=75><br><small>Store</small></td>
<td bgcolor=LightCoral><img src="thumbnails/Street_image_0100.jpg" width=75 height=75><br><small>Street</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/TallBuilding_image_0042.jpg" width=75 height=75><br><small>TallBuilding</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/TallBuilding_image_0051.jpg" width=75 height=75><br><small>TallBuilding</small></td>
</tr>
<tr>
<td>Street</td>
<td>0.620</td>
<td bgcolor=LightBlue><img src="thumbnails/Street_image_0104.jpg" width=75 height=75></td>
<td bgcolor=LightBlue><img src="thumbnails/Street_image_0290.jpg" width=75 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Street_image_0044.jpg" width=75 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Street_image_0042.jpg" width=75 height=75></td>
<td bgcolor=LightCoral><img src="thumbnails/Forest_image_0036.jpg" width=75 height=75><br><small>Forest</small></td>
<td bgcolor=LightCoral><img src="thumbnails/InsideCity_image_0060.jpg" width=75 height=75><br><small>InsideCity</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Street_image_0134.jpg" width=75 height=75><br><small>Street</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Street_image_0006.jpg" width=75 height=75><br><small>Street</small></td>
</tr>
<tr>
<td>Highway</td>
<td>0.830</td>
<td bgcolor=LightBlue><img src="thumbnails/Highway_image_0193.jpg" width=75 height=75></td>
<td bgcolor=LightBlue><img src="thumbnails/Highway_image_0021.jpg" width=75 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Highway_image_0093.jpg" width=75 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Highway_image_0123.jpg" width=75 height=75></td>
<td bgcolor=LightCoral><img src="thumbnails/Street_image_0052.jpg" width=75 height=75><br><small>Street</small></td>
<td bgcolor=LightCoral><img src="thumbnails/Coast_image_0004.jpg" width=75 height=75><br><small>Coast</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Highway_image_0048.jpg" width=75 height=75><br><small>Highway</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Highway_image_0051.jpg" width=75 height=75><br><small>Highway</small></td>
</tr>
<tr>
<td>OpenCountry</td>
<td>0.510</td>
<td bgcolor=LightBlue><img src="thumbnails/OpenCountry_image_0254.jpg" width=75 height=75></td>
<td bgcolor=LightBlue><img src="thumbnails/OpenCountry_image_0116.jpg" width=75 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/OpenCountry_image_0047.jpg" width=75 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/OpenCountry_image_0030.jpg" width=75 height=75></td>
<td bgcolor=LightCoral><img src="thumbnails/Coast_image_0072.jpg" width=75 height=75><br><small>Coast</small></td>
<td bgcolor=LightCoral><img src="thumbnails/Highway_image_0140.jpg" width=75 height=75><br><small>Highway</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/OpenCountry_image_0086.jpg" width=75 height=75><br><small>OpenCountry</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/OpenCountry_image_0017.jpg" width=75 height=75><br><small>OpenCountry</small></td>
</tr>
<tr>
<td>Coast</td>
<td>0.810</td>
<td bgcolor=LightBlue><img src="thumbnails/Coast_image_0123.jpg" width=75 height=75></td>
<td bgcolor=LightBlue><img src="thumbnails/Coast_image_0273.jpg" width=75 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Coast_image_0040.jpg" width=75 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Coast_image_0023.jpg" width=75 height=75></td>
<td bgcolor=LightCoral><img src="thumbnails/OpenCountry_image_0112.jpg" width=75 height=75><br><small>OpenCountry</small></td>
<td bgcolor=LightCoral><img src="thumbnails/OpenCountry_image_0011.jpg" width=75 height=75><br><small>OpenCountry</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Coast_image_0099.jpg" width=75 height=75><br><small>Coast</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Coast_image_0121.jpg" width=75 height=75><br><small>Coast</small></td>
</tr>
<tr>
<td>Mountain</td>
<td>0.840</td>
<td bgcolor=LightBlue><img src="thumbnails/Mountain_image_0138.jpg" width=75 height=75></td>
<td bgcolor=LightBlue><img src="thumbnails/Mountain_image_0139.jpg" width=75 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Mountain_image_0014.jpg" width=75 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Mountain_image_0007.jpg" width=75 height=75></td>
<td bgcolor=LightCoral><img src="thumbnails/Forest_image_0124.jpg" width=75 height=75><br><small>Forest</small></td>
<td bgcolor=LightCoral><img src="thumbnails/Street_image_0149.jpg" width=75 height=75><br><small>Street</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Mountain_image_0110.jpg" width=75 height=75><br><small>Mountain</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Mountain_image_0068.jpg" width=75 height=75><br><small>Mountain</small></td>
</tr>
<tr>
<td>Forest</td>
<td>0.940</td>
<td bgcolor=LightBlue><img src="thumbnails/Forest_image_0129.jpg" width=75 height=75></td>
<td bgcolor=LightBlue><img src="thumbnails/Forest_image_0015.jpg" width=75 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Forest_image_0041.jpg" width=75 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Forest_image_0078.jpg" width=75 height=75></td>
<td bgcolor=LightCoral><img src="thumbnails/Highway_image_0032.jpg" width=75 height=75><br><small>Highway</small></td>
<td bgcolor=LightCoral><img src="thumbnails/Mountain_image_0100.jpg" width=75 height=75><br><small>Mountain</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Forest_image_0033.jpg" width=75 height=75><br><small>Forest</small></td>
<td bgcolor=#FFBB55><img src="thumbnails/Forest_image_0098.jpg" width=75 height=75><br><small>Forest</small></td>
</tr>
<tr>
<th>Category name</th>
<th>Accuracy</th>
<th colspan=2>Sample training images</th>
<th colspan=2>Sample true positives</th>
<th colspan=2>False positives with true label</th>
<th colspan=2>False negatives with wrong predicted label</th>
</tr>
</table>
</center>






