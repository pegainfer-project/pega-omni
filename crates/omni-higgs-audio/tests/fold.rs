use omni_higgs_audio::fold::Fc2;
use omni_higgs_audio::fold::Quantizer;
use omni_higgs_audio::fold::fold;
use omni_kern::weights::Host;
use proptest::prelude::*;

const CODES: usize = 5;
const WIDTH: usize = 3;
const CHANNELS: usize = 4;
const OUT: usize = 2;
const QUANTIZERS: usize = 3;

fn host(shape: &[usize], data: Vec<f32>) -> Host {
    Host { shape: shape.to_vec(), data }
}

fn floats(n: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-1.0f32..1.0, n)
}

proptest! {
    #[test]
    fn a_folded_frame_is_the_quantizers_through_fc2(
        embed in prop::collection::vec(floats(CODES * WIDTH), QUANTIZERS),
        project in prop::collection::vec(floats(CHANNELS * WIDTH), QUANTIZERS),
        project_bias in prop::collection::vec(floats(CHANNELS), QUANTIZERS),
        fc2 in floats(OUT * CHANNELS),
        fc2_bias in floats(OUT),
        frame in prop::collection::vec(0..CODES, QUANTIZERS),
    ) {
        let embeds: Vec<Host> = embed.into_iter().map(|d| host(&[CODES, WIDTH], d)).collect();
        let projects: Vec<(Host, Host)> = project
            .into_iter()
            .zip(project_bias)
            .map(|(w, b)| (host(&[CHANNELS, WIDTH], w), host(&[CHANNELS], b)))
            .collect();
        let (w, b) = (host(&[OUT, CHANNELS], fc2), host(&[OUT], fc2_bias));
        let quantizers: Vec<Quantizer> =
            embeds.iter().zip(&projects).map(|(embed, (w, b))| Quantizer { embed, project_out: (w, b) }).collect();
        let folded = fold(&quantizers, &Fc2(&w, &b));

        let mut sum = vec![0f32; CHANNELS];
        for (q, &code) in frame.iter().enumerate() {
            let (pw, pb) = &projects[q];
            let entry = &embeds[q].data[code * WIDTH..(code + 1) * WIDTH];
            for (c, acc) in sum.iter_mut().enumerate() {
                let row = &pw.data[c * WIDTH..(c + 1) * WIDTH];
                *acc += pb.data[c] + row.iter().zip(entry).map(|(a, b)| a * b).sum::<f32>();
            }
        }
        for (o, (&bias, &fc2_bias)) in folded.bias.iter().zip(&b.data).enumerate() {
            let row = &w.data[o * CHANNELS..(o + 1) * CHANNELS];
            let want = fc2_bias + row.iter().zip(&sum).map(|(a, b)| a * b).sum::<f32>();
            let looked_up: f32 = frame.iter().enumerate().map(|(q, &code)| folded.tables[(q * CODES + code) * OUT + o]).sum();
            prop_assert!((bias + looked_up - want).abs() < 1e-4, "{} against {want}", bias + looked_up);
        }
    }
}
