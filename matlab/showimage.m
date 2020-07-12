function ihout=showimage(img,ah,grayflag,xlim,ylim)

if nargin<2 ah=[]; end
if nargin<3 grayflag=[]; end
if nargin<4 xlim=[1 size(img,2)]; end
if nargin<5 ylim=[1 size(img,1)]; end
if isempty(ah) ah=gca; end
xloc=linspace(xlim(1),xlim(2),size(img,2));
yloc=linspace(ylim(1),ylim(2),size(img,1));

img=double(squeeze(img));
if ~isempty(grayflag) | ndims(img)<3 % show gray image
  gimg=mean(img,3);
  if length(grayflag)>=3
    gmin=grayflag(2);
    gmax=grayflag(3);
  else
    gmin=min(gimg(:));
    gmax=max(gimg(:));
  end
  gres=256;
  range=max(eps,gmax-gmin);
  col=transpose(linspace(0,1,gres));
  ih=image(xloc,yloc,1+gres/range*(gimg-gmin),'Parent',ah);
  set(get(ah,'Parent'),'ColorMap',[col col col]);

else % color image
  minval=min(img(:));
  maxval=max(img(:));
  range=max(eps,maxval-minval);
  img=(img-minval)/range;
  ih=image(xloc,yloc,img,'Parent',ah);
end

if nargout>0 ihout=ih; end
axis(ah,'image');
axis(ah,'off');
