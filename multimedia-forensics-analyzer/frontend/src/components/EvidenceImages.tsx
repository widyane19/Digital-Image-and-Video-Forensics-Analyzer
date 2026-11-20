import type { ImageReport } from '../api'

export function EvidenceImages({ image }: { image: ImageReport }) {
  const { copy_move, compression, edge_texture } = image

  return (
    <section className="card">
      <h3>Pixel analysis</h3>
      <div className="evidence-grid">
        {copy_move && (
          <figure>
            <img src={copy_move.visualization} alt="Image with copy-move candidate regions boxed" />
            <figcaption>
              <strong>Copy-move candidates:</strong>{' '}
              {copy_move.candidate_regions.length === 0
                ? 'no duplicated regions found.'
                : `${copy_move.candidate_regions.length} possible duplicated region pair(s), boxed in green. Repeating patterns can also cause matches, so check them by eye.`}
            </figcaption>
          </figure>
        )}

        {compression && (
          <figure>
            <img src={compression.visualization} alt="Image beside its amplified fine-detail map" />
            <figcaption>
              <strong>Compression detail:</strong> blockiness ratio{' '}
              {compression.blockiness_ratio.toFixed(2)}. Patches with different texture or block
              patterns in the right half can point to pasted or re-saved areas.
            </figcaption>
          </figure>
        )}

        {edge_texture && (
          <figure>
            <img src={edge_texture.edge_map} alt="Edge map of the image" />
            <figcaption>
              <strong>Edges:</strong> density {(edge_texture.edge_density * 100).toFixed(1)}%,
              sharpness {edge_texture.laplacian_variance.toFixed(0)}. Look for edges that are
              unusually sharp or soft compared with their surroundings.
            </figcaption>
          </figure>
        )}
      </div>
    </section>
  )
}
