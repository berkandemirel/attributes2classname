%function analyse()

%normalize = @
r = 'aPaY/GloVe/AWV';

x=load([r,'/trainClasses.mat']);
g_tr = x.trainClasses;
x=load([r,'/testClasses.mat']);
g_te = x.testClasses;

r = 'aPaY/word2vec/300-google';
k = 1;
x=load([r,'/trainClasses.mat']);
w_tr = x.trainClasses * k;
x=load([r,'/testClasses.mat']);
w_te = x.testClasses * k;

% 
% figure(1)
% subplot(1,2,1);
% imagesc(g_tr * g_te.'); title('glove'); colormap(gray)
% subplot(1,2,2);
% imagesc(w_tr * w_te.'); title('w2v'); colormap(gray)

% figure(2)
% subplot(1,2,1);
% imagesc(g_tr * g_tr.'); title('glove'); colormap(gray)
% subplot(1,2,2);
% imagesc(w_tr * w_tr.'); title('w2v'); colormap(gray)


figure(3); clf
subplot(1,2,1)
plot([sum(g_tr.^2,2); sum(g_te.^2,2)]); title('norm-glove');
subplot(1,2,2)
plot([sum(w_tr.^2,2); sum(w_te.^2,2)]); title('norm-w2v');



figure(4); clf
subplot(1,2,1)
plot([mean([g_tr; g_te],1)]); title('mean-glove');
subplot(1,2,2)
plot([mean([w_tr; w_te],1)]); 




figure(5); clf
subplot(1,2,1)
plot([std([g_tr;g_te],1)]);  title('std-glove');
subplot(1,2,2)
plot([std([w_tr;w_te],1)]); 


