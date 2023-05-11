function [xy,xy3] = flyernodes(X,Y)

X = 1; Y = 1;                   % Specify rectangular domain
radius = @radius_square;        % Specify radius function for grain size;
                                % return a radius for each x,y-location
corners = [0,0;X,0;X,Y;0,Y];    % Specify x,y-coordinates for corners of
                                % domain
 

bdyr = discretize_bdy(corners,radius); 

xy = node_drop([-0.05,X+0.05,-0.05,Y+0.05],radius);

xy3 = repel(xy,bdyr,corners,radius);

end
