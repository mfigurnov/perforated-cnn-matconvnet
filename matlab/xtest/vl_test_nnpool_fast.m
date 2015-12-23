function vl_test_nnpool_fast(gpu)

range = 100 ;

if nargin < 1, gpu = false ; end
if gpu
  grandn = @(varargin) range * gpuArray.randn(varargin{:}) ;
else
  grandn = @(varargin) range * randn(varargin{:}) ;
end

x = grandn(15,14,3,2,'single') ;
x(:) = randperm(numel(x))' ;

methods = {'max', 'avg'};
for mi = 1:numel(methods)
  fprintf('testing vl_nnpool \n') ;
  for pool=1:3
    for pad=0:min(3,pool-1)
      for stride=1:4
        args = {'verbose','stride',stride,'pad',pad,'method',methods{mi}};
        idx = vl_nnpoolidx(size(x), pool, args{:});
        if gpu
          idx = gpuArray(permute(idx, [2 3 1]));
        end
        y = vl_nnpoolfast(x,idx,'method',methods{mi}) ;
        y1 = vl_nnpool(x,pool,args{:}) ;
        vl_testsim(y, y1, range * 1e-2);

        dzdy = grandn(size(y),'single') ;
        dzdx = vl_nnpoolfast(x,idx,dzdy,'verbose','method',methods{mi}) ;
        vl_testder(@(x) vl_nnpoolfast(x,idx,'method',methods{mi}), ...
          x, dzdy, dzdx, range * 1e-2) ;
      end
    end
  end

  stride = 1 ;
  pad = 0 ;
  for poolx=1:3
    for pooly=1:2
      pool = [pooly poolx] ;
      args = {'verbose','stride',stride,'pad',pad,'method',methods{mi}};
      idx = vl_nnpoolidx(size(x), pool, args{:});
      if gpu
        idx = gpuArray(permute(idx, [2 3 1]));
      end
      y = vl_nnpoolfast(x,idx,'method',methods{mi}) ;
      y1 = vl_nnpool(x,pool,args{:}) ;
      vl_testsim(y, y1, range * 1e-2);
      dzdy = grandn(size(y),'single') ;
      dzdx = vl_nnpoolfast(x,idx,dzdy,'verbose','method',methods{mi}) ;
      vl_testder(@(x) vl_nnpoolfast(x,idx,'method',methods{mi}), ...
        x, dzdy, dzdx, range * 1e-2) ;
    end
  end

  pool = [3 2] ;
  for stridex=1:3
    for stridey=1:2
      stride = [stridey stridex] ;
      args = {'verbose','stride',stride,'pad',pad,'method',methods{mi}};
      idx = vl_nnpoolidx(size(x), pool, args{:});
      if gpu
        idx = gpuArray(permute(idx, [2 3 1]));
      end
      y = vl_nnpoolfast(x,idx,'method',methods{mi}) ;
      y1 = vl_nnpool(x,pool,args{:}) ;
      vl_testsim(y, y1, range * 1e-2);
      dzdy = grandn(size(y),'single') ;
      dzdx = vl_nnpoolfast(x,idx,dzdy,'verbose','method',methods{mi}) ;
      vl_testder(@(x) vl_nnpoolfast(x,idx,'method',methods{mi}), ...
        x, dzdy, dzdx, range * 1e-2) ;
    end
  end

  pool = [3 4] ;
  stride = [2 1] ;
  for padLeft=0:2
    for padRight=0:2
      pad = [0 0 padLeft padRight] ;
      args = {'verbose','stride',stride,'pad',pad,'method',methods{mi}};
      idx = vl_nnpoolidx(size(x), pool, args{:});
      if gpu
        idx = gpuArray(permute(idx, [2 3 1]));
      end
      y = vl_nnpoolfast(x,idx,'method',methods{mi}) ;
      y1 = vl_nnpool(x,pool,args{:}) ;
      vl_testsim(y, y1, range * 1e-2);
      dzdy = grandn(size(y),'single') ;
      dzdx = vl_nnpoolfast(x,idx,dzdy,'verbose','method',methods{mi}) ;
      vl_testder(@(x) vl_nnpoolfast(x,idx,'method',methods{mi}), ...
        x, dzdy, dzdx, range * 1e-2) ;
    end
  end

  pool = [3 4] ;
  stride = [2 1] ;
  for padTop=0:2
    for padBottom=0:2
      pad = [padTop padBottom 2 1] ;
      args = {'verbose','stride',stride,'pad',pad,'method',methods{mi}};
      idx = vl_nnpoolidx(size(x), pool, args{:});
      if gpu
        idx = gpuArray(permute(idx, [2 3 1]));
      end
      y = vl_nnpoolfast(x,idx,'method',methods{mi}) ;
      y1 = vl_nnpool(x,pool,args{:}) ;
      vl_testsim(y, y1, range * 1e-2);
      dzdy = grandn(size(y),'single') ;
      dzdx = vl_nnpoolfast(x,idx,dzdy,'verbose','method',methods{mi}) ;
      vl_testder(@(x) vl_nnpoolfast(x,idx,'method',methods{mi}), ...
        x, dzdy, dzdx, range * 1e-2) ;
    end
  end

end