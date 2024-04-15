% Paper: Automated Severity Detection Of Chronic Obstructive Pulmonary Disease Using Lung Sounds
% Coder: Arka Roy
% Date: 29/12/2022
% mail id: arka_2121ee34@iitp.ac.in
% Personal Website: https://sites.google.com/view/arka-roy/home?authuser=0&pli=1
%%
clc;
clear all;
close all;
%% Load an example signal
load copd0_labels.mat; load copd0_sigs.mat;
load copd4_labels.mat; load copd4_sigs.mat;
%% VMD decomposition
% some sample parameters for VMD
alpha = 20000;      % moderate bandwidth constraint
tau = 0;            % noise-tolerance (no strict fidelity enforcement)
K = 7;              % 7 modes
DC = 0;             % no DC part imposed
init = 1;           % initialize omegas uniformly
tol = 1e-7;
fs=4000;
sig=copd0_sigs(:,1);s1=sig';
t=0:1/fs:(length(sig)-1)/fs;

[IMF, u_hat, omega] = VMD(s1, alpha, tau, K, DC, init, tol);

figure(1)
subplot(K+1,1,1);plot(t,sig);ylabel('Signal')
for i=2:K+1
    subplot(K+1,1,i);plot(t,IMF(i-1,:));ylabel('IMF'+string(i-1))
    if i==K+1
        xlabel('Time in sec.')
    end
end
%% Feature computation for one signal only
each_fea=[];
for a=1:K
        f=[];
        s=IMF(a,:);
        % features
        mu=mean(s);
        st=std(s);
        ku=kurtosis(s);
        sk=skewness(s);
        mo=mode(s);
        RMS=rms(s);
        E=sum(abs(s).^2);
        se=wentropy(s,'shannon');
        le=wentropy(s,'log energy');
        ApEn=approximateEntropy(s);
        zcr = mean(zerocrossrate(s,WindowLength=0.1*fs,OverlapLength=0.05*fs));
        % stroe them 
        f=[mu,st,ku,sk,mo,RMS,E,se,le,ApEn,zcr];
        each_fea=[each_fea,f];
end
disp('Dimension of feature vector for one signal: '+string(length(each_fea)))

%% Feature computation for all signals from both classes
% just add an outer loop only
all_sigs=[copd0_sigs,copd4_sigs];
master_fea=[];
for i=1:2%length(all_sigs(1,:))
    disp(['iteration==> ' num2str(i)])
    tic
    sig=all_sigs(:,i);s1=sig';
    [IMF, u_hat, omega] = VMD(sig, alpha, tau, K, DC, init, tol);
    each_fea=[];
    for a=1:K
        f=[];
        s=IMF(a,:);
        ku=kurtosis(s);
        sk=skewness(s);
        mu=mean(s);
        st=std(s);
        mo=mode(s);
        E=sum(abs(s).^2);
        se=wentropy(s,'shannon');
        le=wentropy(s,'log energy');
        ApEn=approximateEntropy(s);
        RMS=rms(s);
        zcr = mean(zerocrossrate(s,WindowLength=0.1*fs,OverlapLength=0.05*fs));
        f=[mu,st,ku,sk,mo,RMS,E,se,le,ApEn,zcr];
        each_fea=[each_fea,f];
    end
    master_fea=[master_fea;each_fea];
toc
end
%% Go for ML classification part 
% (I have already saved the 'master fea' variable in matfile you can directly use that)
load master_fea.mat;
all_label=[copd0_labels';copd4_labels'];
[idx_r,weights] = relieff(master_fea,all_label,5);
selected_fea=master_fea(:,idx_r(1:30));
%% Feature ranking using statistical contribution
[idx_r,weights] = relieff(master_fea,all_label,5);
selected_fea=master_fea(:,idx_r(1:30));
%% You can check that which are your final prominant features
j = [ones([11, 1])'  2*ones([11, 1])' 3*ones([11, 1])' 4*ones([11, 1])' 5*ones([11, 1])' 6*ones([11, 1])' 7*ones([11, 1])'];
k=idx_r(1:30);
clc;
disp('=================================================================')
for m=1:length(selected_fea(1,:))

indexed_fea=k(m);

if indexed_fea==1 || indexed_fea==12 || indexed_fea==23||indexed_fea==34||indexed_fea==45||indexed_fea==56||indexed_fea==67
    disp(['Indexed fea ' num2str((indexed_fea)) ': IMF ' num2str(j(indexed_fea)) ' mean'])
elseif indexed_fea==2 ||indexed_fea== 13||indexed_fea==24||indexed_fea==35||indexed_fea==46||indexed_fea==57 || indexed_fea==68
    disp(['Indexed fea ' num2str((indexed_fea)) ': IMF ' num2str(j(indexed_fea)) ' standard dev'])
elseif indexed_fea==3 || indexed_fea==14||indexed_fea==25||indexed_fea==36||indexed_fea==47||indexed_fea==58 ||indexed_fea==69
   disp(['Indexed fea ' num2str((indexed_fea)) ': IMF ' num2str(j(indexed_fea)) ' kurtosis'])
elseif indexed_fea==4 || indexed_fea==15||indexed_fea==26||indexed_fea==37||indexed_fea==48||indexed_fea==59 ||indexed_fea==70
    disp(['Indexed fea ' num2str((indexed_fea)) ': IMF ' num2str(j(indexed_fea)) ' skewness'])    
elseif indexed_fea==5 || indexed_fea==16||indexed_fea==27||indexed_fea==38||indexed_fea==49||indexed_fea==60 ||indexed_fea==71
   disp(['Indexed fea ' num2str((indexed_fea)) ' IMF ' num2str(j(indexed_fea)) ' mode'])    
elseif indexed_fea==6 || indexed_fea==17||indexed_fea==28||indexed_fea==39||indexed_fea==50||indexed_fea==61 ||indexed_fea==72
    disp(['Indexed fea ' num2str((indexed_fea)) ': IMF ' num2str(j(indexed_fea)) ' RMS'])    
elseif indexed_fea==7 || indexed_fea==18||indexed_fea==29||indexed_fea==40||indexed_fea==51||indexed_fea==62 ||indexed_fea==73
    disp(['Indexed fea ' num2str((indexed_fea)) ': IMF ' num2str(j(indexed_fea)) '  Energy'])    
elseif indexed_fea==8  ||indexed_fea== 19||indexed_fea==30||indexed_fea==41||indexed_fea==52||indexed_fea==63 ||indexed_fea==74
   disp(['Indexed fea ' num2str((indexed_fea)) ': IMF ' num2str(j(indexed_fea)) ' Shannon entropy'])    
elseif indexed_fea==9 || indexed_fea==20||indexed_fea==31||indexed_fea==42||indexed_fea==53||indexed_fea==64 ||indexed_fea==75
   disp(['Indexed fea ' num2str((indexed_fea)) ': IMF ' num2str(j(indexed_fea)) ' log energy entropy'])  
elseif indexed_fea==10||indexed_fea==21||indexed_fea==32||indexed_fea==43||indexed_fea==54||indexed_fea==65 ||indexed_fea==76
   disp(['Indexed fea ' num2str((indexed_fea)) ': IMF ' num2str(j(indexed_fea)) ' Approx entropy'])
elseif indexed_fea==11 ||indexed_fea==22||indexed_fea==33||indexed_fea==44||indexed_fea==55||indexed_fea==66 ||indexed_fea==77
    disp(['Indexed fea ' num2str((indexed_fea)) ': IMF ' num2str(j(indexed_fea)) ' ZCR'])

end
end
disp('=================================================================')
%% Split the data in training and testing part
load c.mat;
idxTrain1 = training(c);idxTest=test(c);
train_fea=selected_fea(idxTrain1,:);train_labels=all_label(idxTrain1,:);
test_fea= selected_fea(idxTest,:);test_labels=all_label(idxTest,:);

%% Do a box plot of them
ast_selec=train_fea(1:276,:);health_selec=train_fea(277:554,:); close all;
for i=1:length(train_fea(1,:))
figure(10);
x=[ast_selec(:,i)',health_selec(:,i)'];
G = [ones(size(ast_selec(:,1)))'  2*ones(size(health_selec(:,1)))'];
subplot(5,6,i)
boxplot(x,G,'Labels',{'COPD-0','COPD-4'},'Colors','bk');
find_name=k(i);
if find_name==1 || find_name==12 || find_name==23||find_name==34||find_name==45||find_name==56||find_name==67
    title(['IMF' num2str(j(find_name)) ' mean'])
elseif find_name==2 ||find_name== 13||find_name==24||find_name==35||find_name==46||find_name==57 ||find_name==68
    title(['IMF' num2str(j(find_name)) ' std'])
elseif find_name==3 || find_name==14||find_name==25||find_name==36||find_name==47||find_name==58 ||find_name==69
    title(['IMF' num2str(j(find_name)) ' kurtosis'])
elseif find_name==4 || find_name==15||find_name==26||find_name==37||find_name==48||find_name==59 ||find_name==70
    title(['IMF' num2str(j(find_name)) ' skewness'])    
elseif find_name==5 || find_name==16||find_name==27||find_name==38||find_name==49||find_name==60 ||find_name==71
    title(['IMF' num2str(j(find_name)) ' mode'])    
elseif find_name==6 || find_name==17||find_name==28||find_name==39||find_name==50||find_name==61 ||find_name==72
    title(['IMF' num2str(j(find_name)) ' RMS'])    
elseif find_name==7 || find_name==18||find_name==29||find_name==40||find_name==51||find_name==62 ||find_name==73
    title(['IMF' num2str(j(find_name)) ' Energy'])    
elseif find_name==8  ||find_name== 19||find_name==30||find_name==41||find_name==52||find_name==63 ||find_name==74
    title(['IMF' num2str(j(find_name)) ' Shannon entropy'])    
elseif find_name==9 || find_name==20||find_name==31||find_name==42||find_name==53||find_name==64 ||find_name== 75
    title(['IMF' num2str(j(find_name)) ' Log energy entropy'])  
elseif find_name==10||find_name==21||find_name==32||find_name==43||find_name==54||find_name==65 ||find_name==76
    title(['IMF' num2str(j(find_name)) ' Approx entropy'])
elseif find_name==11 ||find_name==22||find_name==33||find_name==44||find_name==55||find_name==66 ||find_name==77
    title(['IMF' num2str(j(find_name)) ' ZCR'])
end
end
%% use classifier learner to train the ML models using MATLAB

%%
