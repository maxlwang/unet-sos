for i = 0:29
    X = (imread(sprintf(('%d_predict.png'),i)));
    imwrite(ind2rgb(im2uint8(X), parula(256)), (sprintf(('%d_predict_color.png'),i)));
    Y = (imread(sprintf(('%d_truth.png'),i)));
    imwrite(ind2rgb(im2uint8(Y), parula(256)), (sprintf(('%d_truth_color.png'),i)));
    Xsos = double(X)*(300.0/255) + 500.0;
    Ysos = double(Y)*(300.0/255) + 500.0;
    MSE(i+1) = (sumsqr(Xsos-Ysos));
end