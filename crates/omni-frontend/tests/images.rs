use axum::Router;
use axum::body::Body;
use axum::http::Request;
use axum::http::StatusCode;
use base64::Engine as _;
use http_body_util::BodyExt;
use omni_engine::image::Size;
use omni_sim::image::ImageProfile;
use serde_json::Value;
use serde_json::json;
use tower::ServiceExt;

fn app() -> Router {
    let profile = ImageProfile::default();
    let (handle, inbox) = omni_engine::image::channel(profile.info("sim"), 64);
    omni_sim::image::spawn(inbox, profile);
    omni_frontend::router(handle, None)
}

async fn post(app: &Router, body: Value) -> (StatusCode, Value) {
    let req = Request::post("/v1/images/generations")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, serde_json::from_slice(&bytes).unwrap())
}

fn decode(b64: &Value) -> (Size, Vec<u8>) {
    let png = base64::engine::general_purpose::STANDARD.decode(b64.as_str().unwrap()).unwrap();
    let mut reader = png::Decoder::new(std::io::Cursor::new(png)).read_info().unwrap();
    let mut pixels = vec![0; reader.output_buffer_size().unwrap()];
    let frame = reader.next_frame(&mut pixels).unwrap();
    assert_eq!((frame.color_type, frame.bit_depth), (png::ColorType::Rgb, png::BitDepth::Eight));
    pixels.truncate(frame.buffer_size());
    (Size { width: frame.width, height: frame.height }, pixels)
}

#[tokio::test]
async fn every_picture_is_a_png_of_exactly_the_engine_pixels() {
    let app = app();
    let body = json!({ "model": "sim", "prompt": "a red fox", "n": 2, "size": "96x64", "extra": { "seed": 7 } });
    let (status, v) = post(&app, body).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!((v["size"].as_str(), v["output_format"].as_str()), (Some("96x64"), Some("png")));
    let size = Size { width: 96, height: 64 };
    let data = v["data"].as_array().unwrap();
    assert_eq!(data.len(), 2);
    for (i, item) in data.iter().enumerate() {
        let (got, pixels) = decode(&item["b64_json"]);
        assert_eq!(got, size);
        assert_eq!(pixels, omni_sim::image::picture(size, 7, i as u32).pixels.to_vec(), "picture {i}");
    }
}

#[tokio::test]
async fn an_omitted_or_auto_size_is_the_default_one() {
    let app = app();
    for size in [None, Some("auto")] {
        let mut body = json!({ "model": "sim", "prompt": "x", "response_format": "b64_json", "output_format": "png" });
        if let Some(s) = size {
            body["size"] = json!(s);
        }
        let (status, v) = post(&app, body).await;
        assert_eq!(
            (status, v["size"].as_str(), v["data"].as_array().map(Vec::len)),
            (StatusCode::OK, Some("64x64"), Some(1))
        );
    }
}

#[tokio::test]
async fn what_the_server_does_not_produce_is_refused_by_name() {
    let app = app();
    let cases = [
        (json!({ "size": "512x512" }), "size"),
        (json!({ "size": "large" }), "size"),
        (json!({ "n": 0 }), "n"),
        (json!({ "n": 5 }), "n"),
        (json!({ "prompt": "  " }), "prompt"),
        (json!({ "response_format": "url" }), "response_format"),
        (json!({ "output_format": "jpeg" }), "output_format"),
        (json!({ "stream": true }), "stream"),
        (json!({ "extra": { "steps": 4 } }), "extra"),
        (json!({ "extra": { "seed": -1 } }), "extra"),
    ];
    for (patch, param) in cases {
        let mut body = json!({ "model": "sim", "prompt": "x" });
        body.as_object_mut().unwrap().extend(patch.as_object().unwrap().clone());
        let (status, v) = post(&app, body.clone()).await;
        assert_eq!((status, v["error"]["param"].as_str()), (StatusCode::BAD_REQUEST, Some(param)), "{body}");
    }
}

#[tokio::test]
async fn an_unknown_field_or_model_is_an_error_not_a_default() {
    let app = app();
    let (status, v) = post(&app, json!({ "model": "sim", "prompt": "x", "quality": "hd" })).await;
    assert_eq!((status, v["error"]["type"].as_str()), (StatusCode::BAD_REQUEST, Some("invalid_request_error")));
    let (status, v) = post(&app, json!({ "model": "gpt-image-1", "prompt": "x" })).await;
    assert_eq!((status, v["error"]["code"].as_str()), (StatusCode::NOT_FOUND, Some("model_not_found")));
}
