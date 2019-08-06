clear all; close all;
radius=13;
width=6;
N=1000;
TrainSample1=10*N;
distance=-6;
rng(20);
done=0; 
tmp1=[];
TrainSample=N*0.6;
while ~done, 
    tmp=[2*(radius+width/2)*(rand(TrainSample1,1)-0.5) (radius+width/2)*rand(TrainSample1,1)];
    tmp(:,3)=sqrt(tmp(:,1).*tmp(:,1)+tmp(:,2).*tmp(:,2)); 
    idx=find([tmp(:,3)>radius-width/2] & [tmp(:,3)<radius+width/2]);
    tmp1=[tmp1;tmp(idx,1:2)];
    if length(idx)>= TrainSample, 
        done=1;
    end
end
dataTrain=[tmp1(1:TrainSample,:) zeros(TrainSample,1);[tmp1(1:TrainSample,1)+radius -tmp1(1:TrainSample,2)-distance ones(TrainSample,1)]];
R=radius+width/2;

% %plotting the double moon
% figure(1);
% plot(dataTrain(1:TrainSample,1),dataTrain(1:TrainSample,2),'bx',dataTrain(TrainSample+1:end,1),dataTrain(TrainSample+1:end,2),'r+');
% set(gca, 'DataAspectRatio', [1,1,1]);
% title(['Train Sample of Double Moon with Outer radius=' num2str(R) ' width=' num2str(width) ' distance=' num2str(distance) ' Train Sample= ' num2str(TrainSample)]);

%Creating testing data
rng(30);
TestSample=N*0.4;
TestSample1=10*TestSample;
done =0;tmp=[];tmp1=[];
while ~done, 
    tmp=[2*(radius+width/2)*(rand(TestSample1,1)-0.5) (radius+width/2)*rand(TestSample1,1)];
    tmp(:,3)=sqrt(tmp(:,1).*tmp(:,1)+tmp(:,2).*tmp(:,2)); 
    idx=find([tmp(:,3)>radius-width/2] & [tmp(:,3)<radius+width/2]);
    tmp1=[tmp1;tmp(idx,1:2)];
    if length(idx)>= TestSample, 
        done=1;
    end
end
dataTest=[tmp1(1:TestSample,:) zeros(TestSample,1);[tmp1(1:TestSample,1)+radius -tmp1(1:TestSample,2)-distance ones(TestSample,1)]];
X_test=dataTest(:,1:2);
y_test=dataTest(:,3);
R=radius+width/2;
r=R+width;

dataTrain = vertcat(dataTrain,dataTest);
%plotting the double moon
figure(1);
plot(dataTrain(dataTrain(:,3)==1,1),dataTrain(dataTrain(:,3)==1,2),'bx'); hold on;
plot(dataTrain(dataTrain(:,3)==0,1),dataTrain(dataTrain(:,3)==0,2),'r+'); 
%plot(dataTrain(1:TrainSample,1),dataTrain(1:TrainSample,2),'bx',dataTrain(TrainSample+1:end,1),dataTrain(TrainSample+1:end,2),'r+');
set(gca, 'DataAspectRatio', [1,1,1]);
title(['Train Sample Double Moon with Inner Radius = 10, Outer radius=' num2str(R) ' width=' num2str(width) ' distance=' num2str(distance) ' Train Sample=' num2str(TrainSample)]);

%Creating Training matrix
dataTrain=dataTrain';
rng(10);
dataTrain = dataTrain(:,randperm(length(dataTrain)));
final_train_matrix=dataTrain(1:2,:);
target_matrix=dataTrain(3,:);

hidn=3;
learnrate=0.01;
net=feedforwardnet(hidn);
net = configure(net,final_train_matrix,target_matrix);
rng(15);
net = init(net);
net.b{1}=(-0.1 + (2*0.1)*rand(hidn,1));
net.IW{1,1} = (-0.1 + (2*0.1)*rand(hidn,2));
net.trainParam.lr=learnrate; 

%Set maximum epochs to 40 
net.trainParam.epochs=40; 
net.trainParam.min_grad =1e-5;
net.divideParam.trainRatio = 60/100;
net.divideParam.valRatio = 10/100;
net.divideParam.testRatio = 40/100;
weights =[];
[net , tr] = train(net, final_train_matrix, target_matrix);
y2=net(final_train_matrix);
initial_weights = net.IW;
learned_weights = net.lw;

figure(3);
plotperform(tr);
title(['Levenberg–Marquardt Back-Propagation having HiddenNeurons = ' num2str(hidn) 'and Learning Rate = ' num2str(learnrate)]);

out=net(X_test');

figure(4);
plotconfusion(y_test',out);
title('Testing Data for Levenberg–Marquardt Back-Propagation ');
%Plotting non linear plane
figure(6);
cla;
hold on;
margin = 0.05; step = 0.5;
td = X_test';
xlim([min(td(1,:))-margin max(td(1,:))+margin]);
ylim([min(td(2,:))-margin max(td(2,:))+margin]);
bound =0.5;
hold on;
for x = min(td(1,:))-margin : step : max(td(1,:))+margin
   for y = min(td(2,:))-margin : step : max(td(2,:))+margin
   in_td1 = [x y]';
   net_out = net(in_td1);
    if(net_out(1)>=bound)
        plot(x, y, 'y.', 'markersize', 28);  
    elseif (net_out(1)<bound)
        plot(x, y, 'g.', 'markersize', 28);
    end
  end
end
plot(dataTest(dataTest(:,3)==1,1),dataTest(dataTest(:,3)==1,2),'bx'); hold on;
plot(dataTest(dataTest(:,3)==0,1),dataTest(dataTest(:,3)==0,2),'r+'); 
%plot(dataTest(1:TestSample,1),dataTest(1:TestSample,2),'bx',dataTest(TestSample+1:end,1),dataTest(TestSample+1:end,2),'r+');
set(gca, 'DataAspectRatio', [1,1,1]);
title('Decision region for Levenberg–Marquardt Back-Propagation');