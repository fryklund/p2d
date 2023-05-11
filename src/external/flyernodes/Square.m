clear; close all;

X = 1; Y = 1;                   % Specify rectangular domain
radius = @radius_square;        % Specify radius function for grain size;
                                % return a radius for each x,y-location
corners = [0,0;X,0;X,Y;0,Y];    % Specify x,y-coordinates for corners of
                                % domain
                                
ms = 4;                         % Marker size for dots in output figure
mo = 1.6;                       % Marker size for "o"-s in output figure


bdyr = discretize_bdy(corners,radius); 

% Drop nodes of the size given by the function "radius(x,y)" in a slightly
% larger domain
xy = node_drop([-0.05,X+0.05,-0.05,Y+0.05],radius);

figure(1)       % Plot the dropped nodes, as well as the boundary/
                % interface markers
plot(bdyr(:,1),bdyr(:,2),'ko','MarkerSize',mo); hold on
plot(xy(:,1),xy(:,2),'k.','MarkerSize',ms)
axis equal; axis([-0.05,X+0.05,-0.05,Y+0.05])
title('(a)   Boundary points superposed on a 2-D node set') 

% Carry out node repel locally in the vicinity of each boundary /
% interface marker.
xy3 = repel(xy,bdyr,corners,radius);

% Plot the nodes following the repel process.
figure(2)
plot(bdyr(:,1),bdyr(:,2),'ko','MarkerSize',mo); hold on
plot(xy3(:,1),xy3(:,2),'k.','MarkerSize',ms)
axis equal; axis([-0.05,X+0.05,-0.05,Y+0.05])
title('(b)   Node distribution after a local node repel process') 

% print -deps square_nodes.eps