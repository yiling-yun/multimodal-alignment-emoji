function [simmat,D] = simmatrix(inputfeatures,flag,disttype)

if exist('disttype','var')
    D = pdist(inputfeatures,disttype);
else
    D = pdist(inputfeatures);
end;

if exist('flag','var')==0
    flag = 1; 
end;
if flag==1
    simmat = squareform(D);
    simmat(find(eye(size(simmat)))) = mean(D);  % just for display purpose
else
    simmat = squareform(D);
end;