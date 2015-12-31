function vl_test_nnconv_perf(gpu)

range = 100 ;

if nargin < 1, gpu = false ; end
if gpu
  grandn = @(varargin) range * gpuArray.randn(varargin{:}) ;
else
  grandn = @(varargin) range * randn(varargin{:}) ;
end

disp('testing vl_nnconv with square, non square, and fully connected filters') ;
n = 3 ;
fn = 5 ;
for microbatchsize=[1 2]
for bias=[false true]
    for fw=[1 3 5 18]
        for fh=[1 2 3 9]
            w = grandn(fh,fw,10,fn,'single') ;
            if bias
                b = grandn(1,fn,'single') ;
            else
                b = [] ;
            end
            mask = rand([9 18]) >= 0.5;
            maskindices = int32(find(mask(:))) - 1;
            if isempty(maskindices)
              maskindices = int32(0);
            end
            inindices = vl_maskindices_to_outindices(maskindices, [9 18]);
            x = grandn(length(maskindices),1,10,n,'single') ;
            convindices = vl_nnconvidx([9 18 10 n], size(w), 'inindices', inindices, 'verbose');
            if gpu
              convindices = gpuArray(convindices);
            end
            y = vl_nnconv(x,w,b,'convindices',convindices,'verbose') ;
            dzdy = grandn(size(y),'single') ;
            [dzdx,dzdw,dzdb] = vl_nnconv(x,w,b,dzdy,'convindices',convindices,'microbatchsize',microbatchsize,'verbose') ;
            vl_testder(@(x) vl_nnconv(x,w,b,'convindices',convindices,'microbatchsize',microbatchsize), x, dzdy, dzdx, range * 1e-2) ;
            vl_testder(@(w) vl_nnconv(x,w,b,'convindices',convindices,'microbatchsize',microbatchsize), w, dzdy, dzdw, range * 1e-2) ;
            vl_testder(@(b) vl_nnconv(x,w,b,'convindices',convindices,'microbatchsize',microbatchsize), b, dzdy, dzdb, range * 1e-2) ;
        end
    end
end
end

end
