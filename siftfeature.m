% clear all; close all;clc;
% img = imread('dataset/landscape-b.jpg');
function [feature_vec, validpoints] = siftfeature(img)
%% Initialize phase
row = zeros(4,1) ;
col = zeros(4,1);
[row(1), col(1), ch] = size(img);

if(ch == 3)
    new_img = rgb2gray(img);
else
    new_img = img;
end

new_img = double(new_img)./255;
scale_space = cell(5,4);
dog_space = cell(4,4);
sigma_arr = zeros(5,4);

sigma = 1.6;
k = sqrt(2);

%% Create scale space with sigma = 1.6 and k = sqrt(2)
new_img = imresize(new_img, 2, 'bilinear');
for ii=1:size(scale_space,2)
    row(ii) = size(new_img,1);
    col(ii) = size(new_img,2);
    counter = 2*ii - 2;
    for jj=1:size(scale_space,1)
        
        scl = sigma * (k^counter);
        scale_space{jj,ii} = imgaussfilt(new_img, scl);
        sigma_arr(jj,ii) = scl;
        counter = counter + 1;
    end
    new_img = imresize(new_img, 0.5, 'bilinear');
end
%figure(1);title(scale_space);
%% Create DoG space(Difference of Gaussians)
for kk=1:size(dog_space,2)
    for mm=1:size(dog_space,1)
        dog_space{mm,kk} = imsubtract(scale_space{mm+1, kk},scale_space{mm, kk});
    end
end

%% Local Maxima and Minima Detection
% Window size = 3, 3
k = 1;
% First two index for image index, last two for min and max coord.
coord = [];
for aa = 1:size(dog_space,2)
    main2 = dog_space{2, aa};
    main3 = dog_space{3, aa};
    neighof2_1 = dog_space{1, aa};
    neighof2_2 = dog_space{3, aa};
    neighof3_1 = dog_space{2, aa};
    neighof3_2 = dog_space{4, aa};
    
    %Change step size
    for bb = k + 1: 1: row(aa) - k -1
        for cc = k + 1: 1: col(aa) - k - 1
            %Case 1
            main2_window = main2(bb - k: bb + k, cc - k: cc + k);
            neighof2_1window = neighof2_1(bb - k: bb + k, cc - k: cc + k);
            neighof2_2window = neighof2_2(bb - k: bb + k, cc - k: cc + k);
            
            if (main2(bb, cc) == max(main2_window(:))) && (main2(bb, cc) >= max(neighof2_1window(:)))...
                    && (main2(bb, cc) >= max(neighof2_2window(:)))
                
                coord = [coord; 2 aa bb cc sigma_arr(2,aa)];
                
            elseif (main2(bb, cc) == min(main2_window(:))) && (main2(bb, cc) <= min(neighof2_1window(:)))...
                    && (main2(bb, cc) <= min(neighof2_2window(:)))
                
                coord = [coord; 2 aa bb cc sigma_arr(2,aa)];
            end
            
            %Case 2
            main3_window = main3(bb - k: bb + k, cc - k: cc + k);
            neighof3_1window = neighof3_1(bb - k: bb + k, cc - k: cc + k);
            neighof3_2window = neighof3_2(bb - k: bb + k, cc - k: cc + k);
            
            
            if (main3(bb, cc) == max(main3_window(:))) && (main3(bb, cc) >= max(neighof3_1window(:)))...
                    && (main3(bb, cc) >= max(neighof3_2window(:)))
                
                coord = [coord; 3 aa bb cc sigma_arr(3,aa)];
                
            elseif (main3(bb, cc) == min(main3_window(:))) && (main3(bb, cc) <= min(neighof3_1window(:)))...
                    && (main3(bb, cc) <= min(neighof3_2window(:)))
                
                coord = [coord; 3 aa bb cc sigma_arr(3,aa)];
            end
            
        end
    end
end

%Display non-thresholded corner points
disp(size(coord,1));
figure;
% imagesc(img); hold on;
% [xramp,yramp] = meshgrid(1:1:size(dog_space{coord(:,1),coord(:,2)},1),1:1:size(dog_space{coord(:,1),coord(:,2)},1));
% quiver(xramp,yramp,coord(:,4),coord(:,4),10,'r');
% colormap gray;
imshow(img);
hold on;
plot((2.^coord(:,2)-2).*coord(:,4), (2.^coord(:,2)-2).*coord(:,3) , 'b*');

%% Remove non-maximal/minimal outliers
coord2 = [];
% Threshold_value
threshold_1 = 0.03;
for ll=1:size(coord,1)
    
    % Hesssian Matrix Values
    D = dog_space{coord(ll,1), coord(ll,2)};
    Dx = (D(coord(ll,3), coord(ll,4) + 1) - D(coord(ll,3), coord(ll,4) - 1))/2;
    Dy = (D(coord(ll,3) + 1, coord(ll,4)) - D(coord(ll,3) - 1, coord(ll,4)))/2;
    Ds = (dog_space{coord(ll,1) + 1, coord(ll,2)}(coord(ll,3), coord(ll,4))...
        - dog_space{coord(ll,1) - 1, coord(ll,2)}(coord(ll,3), coord(ll,4)))/2;
    Dxx = D(coord(ll, 3), coord(ll, 4) + 1) + D(coord(ll, 3), coord(ll, 4) - 1)...
        - (2*D(coord(ll, 3), coord(ll, 4)));
    Dyy = D(coord(ll, 3) + 1, coord(ll, 4)) + D(coord(ll, 3) - 1, coord(ll, 4))...
        - (2*D(coord(ll, 3), coord(ll, 4)));
    Dxy = (D(coord(ll, 3) + 1, coord(ll, 4) + 1) - D(coord(ll, 3) - 1, coord(ll, 4) + 1)...
        -D(coord(ll, 3) + 1, coord(ll, 4) - 1) + D(coord(ll, 3) - 1, coord(ll, 4) - 1))/4;
    Dxs = (dog_space{coord(ll,1) + 1, coord(ll,2)}(coord(ll, 3), coord(ll, 4) + 1)...
        - dog_space{coord(ll,1) + 1, coord(ll,2)}(coord(ll, 3), coord(ll, 4) - 1)...
        - dog_space{coord(ll,1) - 1, coord(ll,2)}(coord(ll, 3), coord(ll, 4) + 1)...
        + dog_space{coord(ll,1) - 1, coord(ll,2)}(coord(ll, 3), coord(ll, 4) - 1))/4;
    Dys = (dog_space{coord(ll,1) + 1, coord(ll,2)}(coord(ll, 3) + 1, coord(ll, 4))...
        - dog_space{coord(ll,1) + 1, coord(ll,2)}(coord(ll, 3) - 1, coord(ll, 4))...
        - dog_space{coord(ll,1) - 1, coord(ll,2)}(coord(ll, 3) + 1, coord(ll, 4))...
        + dog_space{coord(ll,1) - 1, coord(ll,2)}(coord(ll, 3) - 1, coord(ll, 4)))/4;
    Dss = (dog_space{coord(ll,1) + 1, coord(ll,2)}(coord(ll, 3), coord(ll, 4))...
        + dog_space{coord(ll,1) - 1, coord(ll,2)}(coord(ll, 3), coord(ll, 4)))...
        - (2*D(coord(ll, 3), coord(ll, 4)));
    
    %Clairaut's Theorem: Dxy == Dyx iff. both are continuous within
    %their definition range
    H1 = [Dxx Dxy Dxs; Dxy Dyy Dys; Dxs Dys Dss];
    %Loc_ext => x,y,s
    
    D_dif = [Dx; Dy ;Ds];
    loc_ext = (-pinv(H1)) * D_dif;
    Taylor_dog = D(coord(ll, 3), coord(ll, 4)) + ((D_dif')*loc_ext)/2;
    
    if (abs(Taylor_dog) > threshold_1) && (max(abs(loc_ext)) < 0.5)
        coord2 = [coord2; coord(ll,:) (coord(ll,3)+loc_ext(1)) (coord(ll,4)+loc_ext(2)) (coord(ll,5)+loc_ext(3)) Taylor_dog];
    end
    
end

%Display first_level-thresholded corner points
disp(size(coord2, 1));
plot((2.^coord2(:,2)-2).*coord2(:,4), (2.^coord2(:,2)-2).*coord2(:,3) , 'g+');

%% Eliminate Low Contrast Extremum

coord3 = [];
for ll = 1:size(coord2,1)
    D = dog_space{coord2(ll,1), coord2(ll,2)};
    C_dog = 0.015;
    if (abs(coord2(ll, 9)) >= C_dog) && (abs(D(coord2(ll, 3), coord2(ll, 4))) >=  0.8*C_dog)
        coord3 = [coord3; coord2(ll, :)];
    end
end
disp(size(coord3,1));

%% Eliminate  Edge Responses
coord_final = [];
for ll=1:size(coord3,1)
    
    D = dog_space{coord3(ll,1), coord3(ll,2)};
    Dxx_new = D(coord3(ll, 3), coord3(ll, 4) + 1) + D(coord3(ll, 3), coord3(ll, 4) - 1)...
        - (2*D(coord3(ll, 3), coord2(ll, 4)));
    Dxy_new = (D(coord3(ll, 3) + 1, coord3(ll, 4) + 1) - D(coord3(ll, 3) - 1, coord3(ll, 4) + 1)...
        -D(coord3(ll, 3) + 1, coord3(ll, 4) - 1) + D(coord3(ll, 3) - 1, coord3(ll, 4) - 1))/4;
    Dyy_new = D(coord3(ll, 3) + 1, coord3(ll, 4)) + D(coord3(ll, 3) - 1, coord3(ll, 4))...
        - (2*D(coord3(ll, 3), coord3(ll, 4)));
    
    %Hessian 2x2
    H2 = [Dxx_new Dxy_new; Dxy_new Dyy_new];
    Tr_H2 = H2(1,1) + H2(2,2);
    Det_H2 = (H2(1,1)*H2(2,2)) - (H2(1,2)*H2(2,1));
    
    %Evaluate Edge Responses
    r = 10;
    Ev_H2 = ((Tr_H2)^2)/Det_H2;
    if Ev_H2  < (((r+1)^2)/r)
        coord_final = [coord_final; coord3(ll, :)];
    end
end
disp(size(coord_final,1));
plot((2.^coord_final(:,2)-2).*coord_final(:,4),(2.^coord_final(:,2)-2).*coord_final(:,3) , 'mo');
hold off;
%% Magnitude and Theta Space
mag_space = cell(5,4);
theta_space = cell(5,4);
offset = 50;
for ii = 1:size(scale_space,2)
    for jj=1:size(scale_space,1)
        mag_space{jj, ii} = zeros(size(scale_space{jj,ii},1),size(scale_space{jj,ii},2));
        theta_space{jj, ii} = zeros(size(scale_space{jj,ii},1),size(scale_space{jj,ii},2));
        I = scale_space{jj,ii};
        for xx=2:size(mag_space{jj, ii},1)-1
            for yy=2:size(mag_space{jj, ii},2)-1
                mag_space{jj, ii}(xx,yy) = sqrt(((I(xx + 1, yy)-I(xx -1,yy))^2)...
                    + ((I(xx,yy+1)-I(xx,yy-1))^2));
                theta_space{jj, ii}(xx,yy) = atan2d((I(xx,yy+1)-I(xx,yy-1))...
                    , (I(xx+1,yy)-I(xx-1,yy)));
                theta_space{jj, ii}(xx,yy) = mod(theta_space{jj, ii}(xx,yy) + 360,360);
            end
        end
        mag_space{jj, ii} = padarray(mag_space{jj, ii}, [offset offset],'both');
        theta_space{jj, ii} = padarray(theta_space{jj, ii}, [offset offset],'both');
    end
end
%% Orientation Assignment
coord_final2 = [];
for oo=1:size(coord_final,1)
    hist_bin = zeros(1, 36);
    
    idx = coord_final(oo, 1);
    idy = coord_final(oo, 2);
    mag_I = mag_space{idx, idy};
    theta_I = theta_space{idx, idy};
    %Window size ~ 1.5*sigma
    winsize = 2*ceil(1.5*coord_final(oo,8));
    if mod(winsize,2) == 0
        winsize = winsize + 1;
    end
    gauss_filter = fspecial('gaussian', [winsize winsize], 1.5*coord_final(oo,8));
    currx = coord_final(oo,3) ;%* (2^(coord_final(oo, 2) -idy)); 
    curry = coord_final(oo,4) ;%* (2^(coord_final(oo, 2) -idy));
    
    theta_win = theta_I(currx-(winsize-1)/2+offset:currx+(winsize-1)/2+offset,...
        curry-(winsize-1)/2+offset:curry+(winsize-1)/2+offset);
    mag_win = mag_I(currx-(winsize-1)/2+offset:currx+(winsize-1)/2+offset,...
        curry-(winsize-1)/2+offset:curry+(winsize-1)/2+offset);
    mag_win = mag_win .* gauss_filter;
    for ii = 1:size(theta_win,1)
        for jj = 1:size(theta_win,2)
            theta = mod(theta_win(ii,jj)+360,360);
            m = mag_win(ii,jj);
            hist_bin(1, floor(theta/10)+1) = hist_bin(1, floor(theta/10)+1) + m;
        end
        
    end
    maxEl = max(hist_bin);
    
    for aa = 1:length(hist_bin)
        if hist_bin(aa) >= 0.8*maxEl 
            if aa-1 <= 0
                X = 0:2;
                Y = hist_bin([36,1,2]);
            elseif aa+1 > 36
                X = 35:37;
                Y = hist_bin([35,36,1]);
            else
                X = aa-1:aa+1;
                Y = hist_bin(aa-1:aa+1);
            end
            %Interpolation
            [p,S,mu]  = polyfit([Y(1) Y(2) Y(3)], [X(1) X(2) X(3)], 2); 
            dir = polyval(p,Y(2),[],mu)*10;
            coord_final2 = [coord_final2; coord_final(oo,:) idx idy hist_bin(aa) dir];
        end
    end
end
disp(size(coord_final2,1));

%% Local Image Descriptors
w=4;% In David G. Lowe experiment,divide the area into 4*4.
feature_vec = zeros(size(coord_final2,1),w*w*8);
validpoints = zeros(size(coord_final2,1),2);
for oo = 1:size(coord_final2,1)
    theta_I = theta_space{coord_final2(oo, 10), coord_final2(oo, 11)};
    mag_I = mag_space{coord_final2(oo, 10), coord_final2(oo, 11)};
    
    subpixel_idx = floor(coord_final2(oo, 6));
    subpixel_idy = floor(coord_final2(oo, 7));
    G1 = fspecial('gaussian', [16 16], 8);
    theta = coord_final2(oo,13);
     
    % Lowe descriptor
%     %New approach
%     coor = [subpixel_idx+offset; subpixel_idy+offset];
%     size_mag = [size(mag_I,1)/2; size(mag_I,2)/2];
%     
%     theta = 360 - theta;
%     rotMag = imrotate(mag_I, theta);
%     rotTheta = imrotate(theta_I, theta);
%     
%     rotMat = [cosd(theta) -sind(theta); sind(theta) cosd(theta)];
%     temp_coor = rotMat*(coor - size_mag);
%     rot_coor = temp_coor + size_mag;
%     
%     difMat = [(size(rotMag,1) - size(mag_I,1))/2; (size(rotMag,2) - size(mag_I,2))/2];
%     rot_coor_final = rot_coor + difMat;
%     rot_row = round(rot_coor_final(1));
%     rot_col = round(rot_coor_final(2));
%     
%     temp_vec = zeros(4,4,8);
%     %Create window
%     for i = 1:1:16
%         for j = 1:1:16      
%             rowTrue = round(rot_row + j - 9);
%             colTrue = round(rot_col + i - 9);
%             
%             %Check if points between boundaries
%             if rowTrue > 0 && colTrue > 0 && rowTrue <= size(rotMag, 1) && colTrue <= size(rotMag, 2)
%                 mag = rotMag(rowTrue, colTrue)* G1(j,i);
%                 ori = mod(rotTheta(rowTrue, colTrue) + 360, 360);
%                 temp_vec(ceil(i/4),ceil(j/4),floor(ori/45)+1) = temp_vec(ceil(i/4),ceil(j/4),floor(ori/45)+1) + mag;
%             end
%         end
%     end
%     
%     count = 1;
%     for i = 1:1:4
%         for j = 1:1:4
%             feature_vec(oo,count:count+7) = temp_vec(i,j,:);
%             count = count + 8;
%         end
%     end
    
%AI Shack
    %Current approach
    theta_newwin = theta_I(subpixel_idx-7+offset:subpixel_idx+8+offset,...
        subpixel_idy-7+offset:subpixel_idy+8+offset);
    mag_newwin = mag_I(subpixel_idx-7+offset:subpixel_idx+8+offset,...
        subpixel_idy-7+offset:subpixel_idy+8+offset);
    mag_newwin = mag_newwin .* G1;
    
%     theta_newwin = imrotate(theta_newwin,360-theta,'crop');
%     mag_newwin = imrotate(mag_newwin, 360-theta ,'crop');
    count = 1;
    %For each window, jump 4 by 4
    for i = 1:w:16
        for j = 1:w:16
            theta_f = theta_newwin(i:(i+3), j:(j+3));
            mag_f = mag_newwin(i:(i+3), j:(j+3));
            
            %For each element in window
            for k = 1:4
                for l = 1:4                                    
                    %Rotation dependence
                    theta_bin = theta_f(k,l) - theta + (180/8) + 360;
                    theta_bin = mod(theta_bin,360);
                    
                    %Trilinear interpolation
                    %trilin = 1 - ((((floor(theta_bin/45)*45)-22.5)-theta_bin)/pi);
                    feature_vec(oo, count+floor(theta_bin/45)) = feature_vec(oo, count+floor(theta_bin/45))...
                        + mag_f(k,l);
                    
                end
            end
            count = count + 8;
        end
    end
    
    
    norm = sqrt(sum(feature_vec(oo,:).^2));
    feature_vec(oo,:) = feature_vec(oo,:)./norm;
    
    %Threshold values above 0.2 -> Illumination Independence
    feature_vec(oo, feature_vec(oo,:)>0.2)=0.2;
    
    norm = sqrt(sum(feature_vec(oo,:).^2));
    feature_vec(oo,:) = feature_vec(oo,:)./norm;
    
    validpoints(oo,1) = (2^(coord_final2(oo,2)-2))*coord_final2(oo, 7);
    validpoints(oo,2) = (2^(coord_final2(oo,2)-2))*coord_final2(oo, 6);
    
    
end

% save('feature2.mat','feature_vec');
% save('validpoints2.mat','validpoints');
end
