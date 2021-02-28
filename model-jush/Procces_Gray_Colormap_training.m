for i = 100:199
    X = uint8(255*(imread(sprintf(('./grayscale_training/%d_predict.tif'),i))));
    imwrite(ind2rgb(im2uint8(X), parula(256)), (sprintf(('./color_training/%d_predict_color.png'),i)));
    Y = (imread(sprintf(('./grayscale_training/%d_truth.png'),i)));
    imwrite(ind2rgb(im2uint8(Y), parula(256)), (sprintf(('./color_training/%d_truth_color.png'),i)));
    
    MSE_Keras(i+1) = (sumsqr(double(Y)./255.0 - double(X)./255.0))./(256*256);
    
    Xsos = double(X)*(300.0/255) + 500.0;
    Ysos = double(Y)*(300.0/255) + 500.0;
    
    Error_Map_SoS = Ysos-Xsos;
    imagesc(Error_Map_SoS, [-256 256]), colorbar, cmocean('balance', 'pivot', 0), ax=gca; set(ax, 'xtick',[],'ytick',[]); axis square; H = getframe(gca);  
    %saveas(gcf,sprintf('%d_error.png',i));
    imwrite(H.cdata, sprintf('./color_training/%d_error.png',i));
    %exportgraphics(ax, sprintf('%d_error.png',i));
    
    MSE_SoS(i+1) = (sumsqr(Error_Map_SoS))./(256*256); %Mean Square Error in m/S
    MAE_SoS(i+1) = sum(abs(Error_Map_SoS), 'all')./(256*256);
    
end