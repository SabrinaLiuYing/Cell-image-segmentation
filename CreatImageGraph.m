function NL = CreateImageGraph(U)
    % Get the size of the input image
    [m, n] = size(U);
    mn = m * n;
    
    % Initialize the graph Laplacian matrix
    NL = sparse(mn, mn);

    % Constants for distance and intensity weighting
    sigma2_distance = 150;
    sigma2_intensity = 0.002;
    
    % Convert the image to a column vector for easy indexing
    U = U(:);

    % Construct the graph
    for i = 1:mn
        % Convert 1D index to 2D index
        [x, y] = ind2sub([m, n], i);

        % Initialize the weights for the current pixel
        weights = zeros(1, mn);

        % Consider 8 surrounding neighbor pixels
        for dx = -1:1
            for dy = -1:1
                % Skip the central pixel
                if dx == 0 && dy == 0
                    continue;
                end

                % Compute the 1D index of the neighbor pixel
                nx = x + dx;
                ny = y + dy;
                if nx < 1 || nx > m || ny < 1 || ny > n
                    continue; % Skip if the neighbor is outside the image boundary
                end
                neighbor_idx = sub2ind([m, n], nx, ny);

                % Compute distance and intensity weights
                distance_weight = exp(-(dx^2 + dy^2) / sigma2_distance);
                intensity_weight = exp(-(U(i) - U(neighbor_idx))^2 / sigma2_intensity);

                % Compute the total weight as the product of distance and intensity weights
                total_weight = distance_weight * intensity_weight;

                % Assign the weight to the corresponding entry in the graph Laplacian
                weights(neighbor_idx) = total_weight;
            end
        end

        % Normalize the weights to get the graph Laplacian row
        row_sum = sum(weights);
        if row_sum > 0
            weights = weights / row_sum;
        end

        % Assign the row to the graph Laplacian matrix
        NL(i, :) = weights;
    end

    % Compute the graph Laplacian matrix by subtracting the adjacency matrix from the diagonal matrix
    NL = spdiags(1 - sum(NL, 2), 0, mn, mn) - NL;
end
