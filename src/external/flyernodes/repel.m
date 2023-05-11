function [xy2] = repel(xy,bdyr,corners,radius)

% Carry out the repel process on interior nodes xy that are near any
% boundary or interface marker (as given in the array bdy)

% Output parameter
%   xy2     interior nodes xy after the repel has taken place
    
% Input parameters 
%   xy      interior nodes      Size (:,2) array
%   bdyr    bdy nodes and radii Size (:,3) array
%   corners                     Size (:,2) array
%   radius  radius function 
                    
% Remove any interior nodes that would accidentally be outside the region
% given by "corners"
in = inpolygon(xy(:,1),xy(:,2),corners(:,1),corners(:,2));
xy(~in,:) = [];

% Remove any interior nodes that are within the distance  r/2  of any 
% boundary node. Look for kk nearest neighbors to each bdy point
kk = 6;
[IDX,D] = knnsearch(xy,bdyr(:,1:2),'k',kk);
bdy3 = bdyr(:,3);
ind = find((D-bdy3(:,ones(1,kk))) < 0); 
iremove = IDX(ind);
xy(iremove,:) = [];

[nn,~] = size(xy);  % Find out the number of 'grains' and  
[nb,~] = size(bdyr);% the number of bdy/interface markers

N  = 16;            % Specify the number of nearest neighbors to consider
                    % when repelling; increases each round    
NR =  2;            % Number of 'rounds' (with fresh knnsearch)
NP = 10;            % Number of node pushed within each 'round' (not
                    % updating the identity of 'nearest neighbors').



ne = nn+nb;         % Number of nodes (grains and bdy/interface markers)
xyb = zeros(ne,3);  % Array containing them all; third column will contain
                    % local grain radii.
xyb(1:nn,1:2) = xy;         % First copy interior nodes to this new array,
xyb(nn+1:nn+nb,:) = bdyr;   % then copy the boundary/interface nodes
 
xyb(1:nn,3) = radius(xyb(1:nn,1),xyb(1:nn,2));   % Get radii at all points

ip = 0;             % Count total number of node pushes

for rounds = 1:NR   % Loop over the rounds of node pushing 
                    % Find N nearest neighbors for each pt in xy
    ind  = knnsearch(xyb(:,1:2),xyb(1:nn,1:2),'k',N+1);
    ind(:,1) = [];                      % Remove pointer to itself
                                        % Pointer table size (nn,N)
    ind2 = find(max(ind,[],2)>nn);      % Pointers to elements to be moved;
                    % these are the ones that have a boundary or interface
                    % among its nn nearest neighbors. 
    li2  = length(ind2);                % Number of nodes to be pushed

    % Do the actual node pushing
    cdnn = zeros(N,2);  % Temporary variables
    sdnn = zeros(N,2);
    for k = 1:NP        % Loop over rounds of node pushing
        ip = ip + 1;
        fc = 1/(ip+5);  % Factor with pushing distances each push step
                        % compared to the local grain radius at the point;
                        % the factors are (1/6, 1/7, 1/8, 1/9, ...)
        for np = 1:li2  % Loop over all the nodes that are to be pushed
            xyn  = xyb(ind2(np),:);         % Location of node to push
            xynn = xyb(ind(ind2(np),:),:);  % Collect its nearest neighbors
                        % Find push force; code below has been made
                        % somewhat unreadable in exchange for decent speed;
                        % it implements a repel force of type 1/r^3, then
                        % finds direction of force sum, ignores the
                        % magnitude, and pushes a distance according to 
                        % fc described above.
            cdnn(:,1) = xyn(1)-xynn(:,1); 
            cdnn(:,2) = xyn(2)-xynn(:,2); 
            adnn = 1./sqrt(cdnn(:,1).^2+cdnn(:,2).^2); 
            temp = xynn(:,3).*adnn;
            temp = temp.*temp.*temp.*adnn;        
            sdnn(:,1) = cdnn(:,1).*temp;
            sdnn(:,2) = cdnn(:,2).*temp;
            cn1 = sum(sdnn(:,1));
            cn2 = sum(sdnn(:,2));
            cd = 1/sqrt(cn1^2+cn2^2);
            dir1 = cn1*cd;                      % Direction of push
            dir2 = cn2*cd;
            push1 = fc*xyn(3)*dir1; 
            push2 = fc*xyn(3)*dir2; 
            xyn(1) = xyn(1)+push1;              % Node actually pushed  
            xyn(2) = xyn(2)+push2;
            xyn(3)   = radius (xyn(1),xyn(2));  % Get local grain radius
                                                % at new location
            xyb(ind2(np),:) = xyn;              % xyb array updated
        end
    end
    N = N + 5;
end

xy2 = xyb(1:nn,1:2);    % Place pushed interior nodes in output array
        % On the odd chance that any node would have slipped outside the 
        % overall domain, as defined by 'corners' - eliminate such nodes.
in = inpolygon(xy2(:,1),xy2(:,2),corners(:,1),corners(:,2));
xy2(~in,:) = [];


