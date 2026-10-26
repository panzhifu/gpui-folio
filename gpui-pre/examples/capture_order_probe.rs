use gpui::{
    AtlasTextureId, AtlasTextureKind, AtlasTile, Bounds, CAPTURE_ORDER_BASE, CAPTURE_ORDER_STRIDE,
    ContentMask, DrawOrder, MonochromeSprite, Point, PrimitiveBatch, ScaledPixels, Scene, Size,
    TileId, TransformationMatrix, capture_order_range,
};

fn sprite(order: DrawOrder) -> MonochromeSprite {
    let bounds = Bounds::new(
        Point::new(ScaledPixels(0.0), ScaledPixels(0.0)),
        Size::new(ScaledPixels(1.0), ScaledPixels(1.0)),
    );
    MonochromeSprite {
        order,
        pad: 0,
        bounds,
        content_mask: ContentMask { bounds },
        color: gpui::black(),
        tile: AtlasTile {
            texture_id: AtlasTextureId {
                index: 0,
                kind: AtlasTextureKind::Monochrome,
            },
            tile_id: TileId(0),
            padding: 0,
            bounds: Default::default(),
        },
        transformation: TransformationMatrix::unit(),
    }
}

fn sprite_ranges(scene: &Scene, range: std::ops::Range<DrawOrder>) -> Vec<std::ops::Range<usize>> {
    scene
        .batches_in(range)
        .filter_map(|batch| match batch {
            PrimitiveBatch::MonochromeSprites { range, .. } => Some(range),
            _ => None,
        })
        .collect()
}

fn main() {
    let mut scene = Scene::default();
    for i in 1..=10 {
        scene.monochrome_sprites.push(sprite(i));
    }
    for i in 1..=3 {
        scene
            .monochrome_sprites
            .push(sprite(CAPTURE_ORDER_BASE + CAPTURE_ORDER_STRIDE + i));
    }
    scene.finish();

    let main = sprite_ranges(&scene, 0..CAPTURE_ORDER_BASE);
    let capture = sprite_ranges(&scene, capture_order_range(1));
    println!("main ranges: {main:?}");
    println!("capture ranges: {capture:?}");
    for range in main {
        assert!(range.end <= 10, "主 pass 拿到了捕获段的 sprite: {range:?}");
    }
    for range in capture {
        assert!(
            range.start >= 10,
            "捕获 pass 拿到了主段的 sprite: {range:?}"
        );
    }
}
