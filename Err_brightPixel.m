%img = imread('C:\Users\MEC 101\Desktop\CAM\002\2-5\dot_1.tiff');
%[BW,maskedRGBImage] = MaskFilter(img);
%imshow(maskedRGBImage);

%File location
FileLoc = 'C:\Users\MEC 101\Desktop\CAM\1-11\118\Laserdot\2-5\RedDots\';
FileType = '*.png';
file_list = ScanDir(FileLoc, FileType);

% numebr of dot image:
n_dot = length(file_list);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INITIALIZE %%%%%%%%%%%%%%%%%%%%%%%%%%%
%Matrix to store world value
World_p = zeros(246,2,n_dot);
% initialize max value and its location on image
max_value = zeros(246, 1, n_dot); %finding max value in each dots
Loc =  zeros(246,2, n_dot);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for j = 1: 2 %n_dot: if number of dot image = number of pattern image
    CameraPara = cameraParam;
    Tvecs = CameraPara.TranslationVectors(j,:);
    Rvecs = CameraPara.RotationMatrices(:,:,j);

    imgname = strcat(FileLoc, file_list(j,:));
    img = imread(char(imgname));
    
    imgray = rgb2gray(img);
    thresh = graythresh(imgray);
    imgbinary = imbinarize(imgray, 0.13);% choose 0.1 to conserve all dots
    imgclean = bwareaopen(imgbinary,70); % filter out regoins that has less 70 pixels

    imshow(imgclean);

    List = bwlabel(imgclean); % list of dot pattern
    Label_reg = regionprops(List,'PixelIdxList', 'PixelList'); % lablize and contaon pixel info

    for k = 1:numel(Label_reg)
        [max_value(k),max_idx] = max(imgray(Label_reg(k).PixelIdxList));% maximum value
        Loc(k,:,j) = Label_reg(k).PixelList(max_idx, :);    
    end
%display
    imshow(imgbinary)
    hold on
    plot(Loc(:,1,j),Loc(:,2,j), 'b*')
    hold off

%calcualre back reprojection position via 
    for i = 1: length(Loc)
        World_p(i,:,j) = pointsToWorld(CameraPara,Rvecs, Tvecs, Loc(i,:,j));
    end
end
%calculate for error

%1: calculate the Average error from two directionn for image sets.
Err_xy = zeros(length(Loc),2, n_dot); % initial error set for five images
%2: calculate for average error derivate from the mean position
Err_dots = zeros(length(Loc),1, n_dot);
Loc_ave = mean(World_p,3);
for i = 1:n_dot
    Err_xy(:,:,i) = World_p(:,:,i) - Loc_ave;
    Err_dots(:,:,i) = sqrt(Err_xy(:,1).^2 + Err_xy(:,2).^2);
end
%Average/Max of five
Err_ave = mean(Err_dots);
Err_Max = max(Err_dots);

%sqrt((mean(World_p(:,:,1)- World_p(:,:,2)).^2)/246)


    