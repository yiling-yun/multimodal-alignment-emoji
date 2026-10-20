% alignment model 

clear all;
rng(1);

tic;
datapath = './../data_embeddings/';

knumall = [10 20 50 100 300 600 900];

samplenum = 10000; % 10000; %

flagcluster = 1; % 1: semantic cluster based on fasttext; 2: visual cluster based on VQ

for kci = 1:length(knumall)
    knum = knumall(kci);

    % remove emojies that do not have fasttext embeddings
    sememdfile = '1031_ftr_rep_export.csv';

    stemp = readmatrix([datapath sememdfile]);
    semdraw = stemp(:, 6:305);
    
    idxall = any(~isnan(semdraw),2);
    idxidx = find(idxall==1);
    semdall = semdraw(idxall,:); 
    
    if flagcluster == 2
        vememdfile = '1031_ae_rep_export.csv';
        vtemp = readmatrix([datapath vememdfile]);
        vraw = vtemp(:, 6:305);
        semdall = vraw(idxall,:); 
    end;
    
    [idx_cos, V, D] =  spectralcluster(semdall, knum, 'Distance', 'cosine');
    [gc,grps] = groupcounts(idx_cos);
    
    % find the item that is closest to centroid for each cluster
    for i = 1:knum
        idxvec = [];
        idxvec = find(idx_cos==i);
        meanvec = mean(semdall(idxvec,:),1);
        disttemp = squareform(pdist([meanvec;semdall(idxvec,:)],'cosine'));
        [~,minidx]=min(disttemp(1,2:end));
        idxselk(i) = idxvec(minidx);
    end;
    idxalltemp = idxidx(idxselk);
    clear idxall;
    idxall = idxalltemp;
    
    
    clear sememdfile stemp semdraw;
    
    %% sim 1
    simlabel = 'sim1';
    visemdfile = '1031_ae_rep_export.csv';
    sememdfile = '1031_ftr_rep_export.csv';
    
    vtemp = readmatrix([datapath visemdfile]);
    vemdraw = vtemp(:, 6:end);
    
    stemp = readmatrix([datapath sememdfile]);
    semdraw = stemp(:, 6:end);
    
    vemd= vemdraw(idxall,:); 
    semd= semdraw(idxall,:); 
    
    itemnum = size(semd,1);
    
    
    
    mappingaccu = [0:0.1:1];
    
    
    for mi = 1:length(mappingaccu)
        for si = 1:samplenum
            V_vec = []; S_vec = []; indx1 = []; indx2 =[];
    
            vecshuff1 = randperm(itemnum);
            V_vec = vemd(vecshuff1,:);
    
            if mi>1
                mapdim = round(mappingaccu(mi)*itemnum);
                indx1 = vecshuff1(1:mapdim);
                
                vecshuff2 = randperm(itemnum-mapdim)+mapdim;
                indx2 = vecshuff1(vecshuff2);            
                
                S_vec = [semd(indx1,:); semd(indx2,:)];
    
            else
                S_vec = semd(randperm(itemnum),:);
            end;
    
    
            [simmatV, simmatVDs] = simmatrix(V_vec,0,'cosine');
    
            [simmatS, simmatSDs] = simmatrix(S_vec,0,'cosine');
    
    
    
            % correlation
            data = [simmatVDs' simmatSDs' ];
            rsttemp = corr(data,'Type','Spearman');
            rstcorr(mi,si) = rsttemp(1,2);
        end;
    
    end;
    toc;
    
    meanrstcorr(:,kci) = mean(rstcorr,2);
    
    aligncorrthreshold = rstcorr(length(mappingaccu),1);
    
    for mi = 1:length(mappingaccu)-1
        propovernum(mi,1) = sum(rstcorr(mi,:)>=aligncorrthreshold)/samplenum;
    end
    
    meanprop(1,kci) = mean(propovernum);
    save([ '.\rst\rst1_' simlabel  '.' num2str(knum) '.' num2str(flagcluster) '.mat'],'rstcorr','visemdfile','sememdfile',...
        'meanrstcorr','meanprop');


end;

