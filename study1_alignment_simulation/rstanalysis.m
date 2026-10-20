% data analysis for alignment correlation and mapping accuracy

clear all;


knumall = [10 20 50 100 300 600 900];
flagcluster = 1; % 1: semantic cluster based on fasttext; 2: visual cluster based on VQ
figure;

for ki = 1:length(knumall)
    subplot(2,4,ki);
    conceptnum = knumall(ki); 
    
    % filename = ['rst_sim1.' num2str(conceptnum) '.' num2str(flagcluster)];
    % mappingaccu = [0:0.1:1];
   
    filename = ['rst1_sim1.' num2str(conceptnum) '.' num2str(flagcluster)];
    mappingaccu = [0:0.02:1];

    load(['./rst/' filename '.mat']);
    
    samplenum = size(rstcorr,2); %1000; % 
    
    aligncorrthreshold = rstcorr(length(mappingaccu),1);
    
    for mi = 1:length(mappingaccu)
        y(mi) = mappingaccu(mi);
        dtemp = rstcorr(mi,:);
        x(mi) = mean(dtemp);
        stdval(mi) = std(dtemp);
        minval(mi) = min(dtemp);
        maxval(mi) = max(dtemp);
    
    end
    [~,indx]= min(abs(y-(1-1/conceptnum)));
    if isempty(indx)
        indx = conceptnum;
    end;
    xx = x(1:indx); %reshape(rstcorr,1,[]);  % 
    yy = y(1:indx); %kron(y,ones(1,samplenum));  % 
    [rho(ki),pval(ki)] = corr(xx',yy');
    
    
    yconf = [y y(end:-1:1)] ;         
    xconf = [x+stdval x(end:-1:1)-stdval(end:-1:1)];
    xrange = [minval maxval(end:-1:1)];

    p1 = fill(xrange,yconf,'red');
    p1.FaceColor = [1 0.8 0.8];      
    p1.EdgeColor = 'none';  
    
    hold on
    p = fill(xconf,yconf,'red');
    p.FaceColor = [1 0.8 0.8]*0.8;      
    p.EdgeColor = 'none';   
    
    hold on
    indx = maxval>aligncorrthreshold;
    maxval2 = indx.*maxval + (1-indx).*aligncorrthreshold;
    xrange2 = [aligncorrthreshold*ones(size(x)) maxval2(end:-1:1)];
    
    p2 = fill(xrange2,yconf,'red');
    p2.FaceColor = [1 0 0];      
    p2.EdgeColor = 'red';   
    
    hold on
    plot(x,y,'b-')
    hold off
    
    axis square;
    title([num2str(conceptnum) ' Emojis'], 'FontSize', 20, 'FontWeight', 'bold');

    % Make all axis tick labels larger
    set(gca, 'FontSize', 20, 'LineWidth', 1.5)
   
end;
han = axes(gcf,'visible','off'); 
han.XLabel.Visible = 'on';
han.YLabel.Visible = 'on';
xlabel('Alignment correlation of similarity', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Mapping accuracy', 'FontSize', 20, 'FontWeight', 'bold');

meanprop
[rho; pval] 