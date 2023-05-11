function xy = node_drop (Box, radius)

% General node dropping code for the case of spatially varying grain raddii

% Output parameter
%   xy      Matrix xy(:,2) with generated node locations
%
% Input parameters
%   Box         Vector containing [Xmin, Xmax, Ymin, Ymax], where
%               the entries specify the rectangle that is to
%               be filled with nodes
%   radius      The function radius(x,y) provides grain radius to be used
%               at location (x,y).

%   Initialization - specify some upper limits for internal array sizes
ninit  = 5e+3;          % Max allowed number of potential dot positions 
dotmax = 5e+5;          % Max allowed number of final dots

dotnr  = 0;             % Count the placed dots

pdp      = zeros(ninit,2);  % Array to hold potential dot positions  (pdp)
                        % Place initial pdps along bottom boundary, with a
                        % spatially variable density consistent with the
                        % local grain radii.
Xmin = Box(1); Xmax = Box(2); Ymin = Box(3); Ymax = Box(4);                        
xy_bottom = discretize_bdy([Xmin,Ymin;Xmax,Ymin], radius);

pdp_end = length(xy_bottom(:,1));   % The total number of markers that are 
                                    % in use at any time
pdp(1:pdp_end,:) = xy_bottom(:,1:2);% Copy over marker positions from
                                    % xy_bottom to array holding potential
                                    % dot positions (PDP).
xy       = zeros(dotmax,2);         % Array to store produced actual dot
                                    % locations

[ym,i]   = min(pdp(1:pdp_end,2));   % Lowest PDP y-value, and its potition

while ym <= Ymax && dotnr < dotmax; % Loop over all dots to be placed
    dotnr = dotnr + 1;      % Keep count on number of next dot to be placed
    xy(dotnr,:) = pdp(i,:); % Place the dot

    r = radius(xy(dotnr,1),xy(dotnr,2)); % Get grain radius to be used 
                                    % at this dot location
    
    % Calculate the distance from the placed dot to all present markers
    dist2 = (pdp(1:pdp_end,1)-pdp(i,1)).^2+(pdp(1:pdp_end,2)-pdp(i,2)).^2;
    
    ileft  = find(dist2(1:i)  > r^2,1, 'last' );  % Find nearest marker
                                    % to the left, outside the circle
    if isempty(ileft );             % Special case if hitting left boundary
        ileft  = 0;                 
        ang_left = pi;
    else
        il =      ileft;
        ang_left  = atan2(pdp(il,2)-pdp(i,2),pdp(il,1)-pdp(i,1));
    end
    iright = find(dist2(i:pdp_end) > r^2,1, 'first'); % Find nearest marker
                                    % to the right, outside the circle
    if isempty(iright);             % Special case if hitting right boundary
        iright = 0;             
        ang_right = 0;
    else
        ir = i + iright -1;
        ang_right = atan2(pdp(ir,2)-pdp(i,2),pdp(ir,1)-pdp(i,1));
    end
    
    % Introduce new markers along circle sector, equispaced in angle
    % between the directions ang_left and ang_right
    ang = ang_left-[0.1;0.2;0.3;0.4;0.5;0.6;0.7;0.8;0.9]*(ang_left-ang_right);        
    pdp_new = [pdp(i,1)+r*cos(ang),pdp(i,2)+r*sin(ang)];
    ind = pdp_new(:,1) < Xmin | pdp_new(:,1) > Xmax; % Check if any new 
    pdp_new(ind,:) = [];        % markers outside the domain; if so, remove
    
    nw = length(pdp_new(:,1));  % Number of new markers to be inserted
    
    % Find and remove entries in pdp that the last dot have made obsolete
    if iright ~= 0;                    % Identify block to the right
        nr_right     = pdp_end-i-iright+2;
        pdp_right    = pdp(i+iright-1:pdp_end,:);
    end
 
    pdp(ileft+1:ileft+nw,:) = pdp_new;  % Insert the new markers into pdp
    
    if iright == 0              % Place rightmost block (of old markers)
        pdp_end = ileft+nw;     % to the right of the block of new markers
    else
        pdp_end = ileft+nw+nr_right;
        pdp(ileft+nw+1:pdp_end,:) = pdp_right;
    end
   
    [ym,i] = min(pdp(1:pdp_end,2));     % Identify next dot location 
end                             % Return to top of loop

xy = xy(1:dotnr,:); % Return to the calling program just the actually
                    % placed dots.
