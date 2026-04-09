//
//  Models.swift
//  SommelierAI
//
//  PURPOSE
//  Canonical domain models for SommelierAI iOS layer.
//  Single source of truth for wine/search data structures.
//  Used by ViewModels, UI rendering and API decoding.
//
//  SCOPE
//  - SearchRequest
//  - SearchResponse
//  - SearchMeta
//  - WineCard
//
//  OUT OF SCOPE
//  - SwiftUI Views
//  - SSE transport envelopes (APIStreamEvent)
//  - Networking logic (APIClient)
//
//  RULES
//  - No duplicate type definitions elsewhere.
//  - Robust decoding (backend may send string or number).
//  - Debug fields must remain optional.
//  - This file contains ONLY domain models.
//

import Foundation

// MARK: - SearchRequest

struct SearchRequest: Codable {
    var query: String
    var user_prefs: [String: String]?
    var sort_mode: String
}

// MARK: - SearchResponse

struct SearchResponse: Codable {
    var results: [WineCard]
    var meta: SearchMeta?
}

// MARK: - SearchMeta

struct SearchMeta: Codable {
    var query: String?
    var sort: String?
    var limit: Int?
    var offset: Int?
    var total: Int?
    var total_count: Int?  // Totale vini disponibili (per paginazione)
    var rank_mode: String?
    var model: String?
    var build_id: String?
}

// MARK: - AromaIcon

struct AromaIcon: Codable, Hashable {
    var name: String
    var icon: String
}

// MARK: - WineCard

struct WineCard: Identifiable, Codable, Hashable {

    // MARK: Identity

    var id: String
    var name: String
    var reason: String

    // MARK: Ranking

    var rank: Int?
    var score: Double?
    var match_score: Double?
    var __match_score: Double?
    var __quality_score: Double?
    var __value_score: Double?
    var popularity: Double?

    // MARK: Price

    var price: Double?
    var purchase_url: String?

    // MARK: Origin

    var producer: String?
    var country: String?
    var region: String?
    var zone: String?
    var denomination: String?
    var vintage: Int?
    var color: String?

    // MARK: Descriptors

    var grapes: String?
    var intensity: String?
    var sparkling: String?             // ✅ NEW: tipo frizzante (fermo/spumante/frizzante)
    var freshness: String?             // ✅ NEW: freschezza (low/medium/high)
    var sweetness: String?             // ✅ NEW: dolcezza (secco/abboccato/dolce)
    var food_pairings: [String]?
    var tags: [String]?
    var aromas: [String]?              // ✅ NEW: aromi derivati da backend
    var aroma_icons: [AromaIcon]?      // ✅ NEW: icone aromi con emoji
    var ottimo_valore: Bool?           // ✅ NEW: badge "Ottimo Valore" (LEGACY)
    var show_value_badge: Bool?        // ✅ NEW: badge "Ottimo Valore" (NUOVO)
    var tasting_notes: String?         // ✅ NEW: note degustazione LLM (lazy load)
    var reviews_count: Int?            // ✅ NEW: numero recensioni (mock)
    var critic_score: Int?             // ✅ NEW: punteggio critico (mock)

    // MARK: Ratings

    var rating_overall: Double?

    // MARK: Explainability

    var ui_highlights: [String]?
    var match_explanation: [String]?   // ✅ NEW: spiegazione match (array di stringhe)
    var match_breakdown: [String: Double]?
    var explain: [String]?
    var __components: [String: Double]?

    // MARK: Qualitative

    var quality: String?
    var balance: String?
    var persistence: String?
    var color_detail: String?

    // MARK: CodingKeys

    private enum CodingKeys: String, CodingKey {
        case id, name, reason
        case rank, score
        case match_score
        case __match_score
        case __quality_score
        case __value_score
        case popularity
        case price, purchase_url
        case producer, country, region, zone, denomination
        case vintage
        case color
        case grapes, intensity
        case sparkling, freshness, sweetness  // ✅ NEW
        case food_pairings
        case tags
        case aromas              // ✅ NEW
        case aroma_icons         // ✅ NEW
        case ottimo_valore       // ✅ NEW (LEGACY)
        case show_value_badge    // ✅ NEW
        case tasting_notes       // ✅ NEW
        case reviews_count       // ✅ NEW
        case critic_score        // ✅ NEW
        case rating_overall
        case ui_highlights
        case match_explanation   // ✅ NEW
        case match_breakdown
        case quality, balance, persistence, color_detail
        case explain
        case __components
    }

    // MARK: - Custom Decoder (robust)

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)

        id = try c.decodeIfPresent(String.self, forKey: .id) ?? UUID().uuidString
        name = try c.decodeIfPresent(String.self, forKey: .name) ?? ""
        reason = try c.decodeIfPresent(String.self, forKey: .reason) ?? ""

        rank = try c.decodeIfPresent(Int.self, forKey: .rank)
        score = Self.decodeDoubleRelaxed(c, .score)

        match_score = Self.decodeDoubleRelaxed(c, .match_score)
        __match_score = Self.decodeDoubleRelaxed(c, .__match_score)
        __quality_score = Self.decodeDoubleRelaxed(c, .__quality_score)
        __value_score = Self.decodeDoubleRelaxed(c, .__value_score)
        popularity = Self.decodeDoubleRelaxed(c, .popularity)

        price = Self.decodeDoubleRelaxed(c, .price)
        purchase_url = try c.decodeIfPresent(String.self, forKey: .purchase_url)

        producer = try c.decodeIfPresent(String.self, forKey: .producer)
        country = try c.decodeIfPresent(String.self, forKey: .country)
        region = try c.decodeIfPresent(String.self, forKey: .region)
        zone = try c.decodeIfPresent(String.self, forKey: .zone)
        denomination = try c.decodeIfPresent(String.self, forKey: .denomination)
        vintage = Self.decodeIntRelaxed(c, .vintage)
        color = try c.decodeIfPresent(String.self, forKey: .color)

        grapes = try c.decodeIfPresent(String.self, forKey: .grapes)
        intensity = try c.decodeIfPresent(String.self, forKey: .intensity)
        sparkling = try c.decodeIfPresent(String.self, forKey: .sparkling)  // ✅ NEW
        freshness = try c.decodeIfPresent(String.self, forKey: .freshness)  // ✅ NEW
        sweetness = try c.decodeIfPresent(String.self, forKey: .sweetness)  // ✅ NEW

        food_pairings = Self.decodeStringListRelaxed(c, .food_pairings)
        tags = Self.decodeStringListRelaxed(c, .tags)
        aromas = try c.decodeIfPresent([String].self, forKey: .aromas)  // ✅ NEW
        aroma_icons = try c.decodeIfPresent([AromaIcon].self, forKey: .aroma_icons)  // ✅ NEW
        ottimo_valore = try c.decodeIfPresent(Bool.self, forKey: .ottimo_valore)  // ✅ NEW (LEGACY)
        show_value_badge = try c.decodeIfPresent(Bool.self, forKey: .show_value_badge)  // ✅ NEW
        tasting_notes = try c.decodeIfPresent(String.self, forKey: .tasting_notes)  // ✅ NEW
        reviews_count = try c.decodeIfPresent(Int.self, forKey: .reviews_count)  // ✅ NEW
        critic_score = try c.decodeIfPresent(Int.self, forKey: .critic_score)  // ✅ NEW

        rating_overall = Self.decodeDoubleRelaxed(c, .rating_overall)

        ui_highlights = try c.decodeIfPresent([String].self, forKey: .ui_highlights)
        match_explanation = try c.decodeIfPresent([String].self, forKey: .match_explanation)  // ✅ NEW
        explain = Self.decodeStringListRelaxed(c, .explain)
        match_breakdown = try c.decodeIfPresent([String: Double].self, forKey: .match_breakdown)

        quality = try c.decodeIfPresent(String.self, forKey: .quality)
        balance = try c.decodeIfPresent(String.self, forKey: .balance)
        persistence = try c.decodeIfPresent(String.self, forKey: .persistence)
        color_detail = try c.decodeIfPresent(String.self, forKey: .color_detail)
        
        // ✅ ROBUST: Ignora __components se il formato non corrisponde (debug-only field)
        __components = try? c.decodeIfPresent([String: Double].self, forKey: .__components)

        // ✅ Fallbacks (requested)
        // 1) Match: use whichever field is present to avoid fake 0% in UI.
        if match_score == nil, let m = __match_score { match_score = m }
        if __match_score == nil, let m = match_score { __match_score = m }

        // 2) Rating: if backend doesn't provide rating_overall (or it's 0), align it to score.
        if let s = score, (rating_overall ?? 0) == 0 {
            rating_overall = s
        }
    }

    // MARK: - Relaxed Decoders

    private static func decodeDoubleRelaxed(
        _ c: KeyedDecodingContainer<CodingKeys>,
        _ key: CodingKeys
    ) -> Double? {
        if let v = try? c.decodeIfPresent(Double.self, forKey: key) { return v }
        if let s = try? c.decodeIfPresent(String.self, forKey: key) {
            let cleaned = s
                .replacingOccurrences(of: ",", with: ".")
                .trimmingCharacters(in: CharacterSet.whitespacesAndNewlines)
            return Double(cleaned)
        }
        return nil
    }

    private static func decodeIntRelaxed(
        _ c: KeyedDecodingContainer<CodingKeys>,
        _ key: CodingKeys
    ) -> Int? {
        if let v = try? c.decodeIfPresent(Int.self, forKey: key) { return v }
        if let s = try? c.decodeIfPresent(String.self, forKey: key) {
            return Int(s.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines))
        }
        return nil
    }

    private static func decodeStringListRelaxed(
        _ c: KeyedDecodingContainer<CodingKeys>,
        _ key: CodingKeys
    ) -> [String]? {
        if let arr = try? c.decodeIfPresent([String].self, forKey: key) {
            return arr.filter {
                !$0.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines).isEmpty
            }
        }
        if let s = try? c.decodeIfPresent(String.self, forKey: key) {
            let parts = s
                .components(separatedBy: CharacterSet(charactersIn: "|;,"))
                .map { $0.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
            return parts.isEmpty ? nil : parts
        }
        return nil
    }
}
