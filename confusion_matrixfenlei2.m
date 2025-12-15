function confusion_matrix1(act1,det1)
[mat,order] = confusionmat(act1,det1);
k=length(order);             %k denotes the number of categories
mat1 = repmat(sum(mat,2),1,k);
mat =mat./mat1;

figure('color',[1 1 1]),imagesc(mat); %# Create a colored plot of the matrix values
colormap(flipud(gray));  %# Change the colormap to gray (so higher values are

%#black and lower values are white)
title('Confusion Matrix'); 
textStrings = num2str(mat(:).*100,'%0.02f%%');       %# Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));  %# Remove any space padding

%% ################
%# Create x and y coordinates for the strings %meshgrid is a function in MATLAB used to generate grid sampling points
[x,y] = meshgrid(1:k);  
hStrings=text(x(:),y(:),textStrings(:),'HorizontalAlignment','center','FontSize',16);
midValue = mean(get(gca,'CLim'));  %# Get the middle value of the color range
textColors = repmat(mat(:) > midValue,1,3);  %# Choose white or black for the
                                             %#   text color of the strings so
                                             %#   they can be easily seen over
                                             %#   the background color
set(hStrings,{'Color'},num2cell(textColors,2));  %# Change the text colors£»

set(gca,'XTick',1:10,...                                    
        'XTickLabel',{'Live','apoptosis','necrosis'},...  %#   and tick labels
        'YTick',1:10,...                                    
        'YTickLabel',{'Live','apoptosis','necrosis'},...
        'TickLength',[0 0],'FontName','Times New Roman','FontSize',16);
%==========================================================    
