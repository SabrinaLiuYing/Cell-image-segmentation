%
%
%   Cell image segmentation
%


%
% Read in a block from a cell image
%
U = imread('cellimage.tif');
U = U(90:190,190:290);
U = double(U);
U = U/max(U(:));


%
% Create the normalized graph Laplacian from image U
% You will need to implement this function for ***part (a).***

NL = CreateImageGraph(U);

%
% Perform normalized spectral clustering
%

%*** Provide your code here for part (b)! ***


% Perform eigenvalue decomposition to get the eigenvectors and eigenvalues
K = 9; 
[V, D] = eigs(NL, K, 'sa');
replicates = 20;
index = kmeans(normr(V), K, 'replicates', replicates);


%result should be the variable 'index' produced by Matlabs kmeans command, 
%i.e., a vector of length m*n containing the cluster index for each pixel



%
% Extract segments for the expected cell region in a simple way
%
Clusters = reshape(index,size(U,1),size(U,2));

Disk = fspecial('disk',floor(size(U,1)/2));
Disk = Disk>0;

Cell = zeros(size(U));
for k=1:K
    seg_size = nnz(Clusters==k);
    overlap = (Clusters==(Disk*k));
    in_size = nnz(overlap);
    if in_size == seg_size,
        Cell = Cell + (Clusters==k);
    end
end
Cell = 2*(Cell-0.5);


%
% Visualize segmentation results
%
figure(1);

%input image
subplot(1,3,1);
imshow(U,[]);

%generated clusters
subplot(1,3,2);
imshow(Clusters,[]);

%segmented result
subplot(1,3,3);
imshow(U,[]);
hold on;
contour(Cell,[0 0],'r', 'linewidth', 1.5);
hold off;

function NL = CreateImageGraph(U)
    % initialize
    [m, n] = size(U);
    mn = m * n;
    NL = sparse(mn, mn);
    sigma2_dis = 150;
    sigma2_int = 0.002;
    U = U(:);

    for i = 1:mn
        [x, y] = ind2sub([m, n], i);
        weights = zeros(1, mn);

        % Consider 8 surrounding neighbor pixels
        for ii = -1:1
            for jj = -1:1

                % skip central
                if ii == 0 && jj == 0
                    continue;
                end

                % skip outside
                if x+ii < 1 || x+ii > m || y + jj < 1 || y + jj > n
                    continue; 
                end

                neighbor = sub2ind([m, n], x+ii, y + jj);
                weights(neighbor) = exp(-(ii^2 + jj^2) / sigma2_dis) * exp(-((U(i) - U(neighbor))^2) / sigma2_int);
            end
        end

        % Normalize the weights 
        row_sum = sum(weights);
        if row_sum > 0
            weights = weights / row_sum;
        end
        NL(i, :) = weights;
    end

    % subtracting the adjacency matrix from the diagonal matrix
    NL = spdiags(1 - sum(NL, 2), 0, mn, mn) - NL;
end

