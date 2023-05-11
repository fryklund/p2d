function r = radius_square(x,y)
d  = min([abs(x-0),abs(x-1),abs(y-0),abs(y-1)],[],2);
r  = 0.045+0.045*atan(12*d);   % On handout
%r  = 0.002+0.01*atan(6*d);    % Closer to Ellitic PDE paper node layout
end
