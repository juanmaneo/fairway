use fairway_hash::{fairway128, fairway256, fairway64, sip_hash, Key, SipHashKey};

fn main() {
    let key = Key([1, 2, 3, 4]);
    let data = b"The quick brown fox jumps over the lazy dog";

    println!("backend: {}", fairway_hash::best_backend().name());
    println!("fairway64:  {:016x}", fairway64(key, data));
    println!(
        "fairway128: {:016x}{:016x}",
        fairway128(key, data)[1],
        fairway128(key, data)[0]
    );
    let h256 = fairway256(key, data);
    println!(
        "fairway256: {:016x}{:016x}{:016x}{:016x}",
        h256[3], h256[2], h256[1], h256[0]
    );

    let sip_key = SipHashKey([0x0706050403020100, 0x0F0E0D0C0B0A0908]);
    println!("siphash-2-4:    {:016x}", sip_hash(sip_key, data));
}
