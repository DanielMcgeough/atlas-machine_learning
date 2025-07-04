-- lists all bands with Glam rock as their main style, ranked by their longevity (as of 2020)
SELECT band_name, IFNULL(split, 2020) - formed AS lifespan
FROM metal_bands
WHERE style LIKE "%Glam rock%"
ORDER BY lifespan DESC;
