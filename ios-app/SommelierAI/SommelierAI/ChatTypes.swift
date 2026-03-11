//
//  ChatTypes.swift
//  SommelierAI
//
//  PURPOSE
//  Chat domain types: message structures, roles, and chat-only state.
//  Defines how the conversation is represented inside the app.
//
//  SCOPE (OWNED TYPES)
//  - ChatDomain
//  - ChatDomain.Message
//  - Chat-only enums/structs (roles, message kinds, local chat state)
//
//  OUT OF SCOPE (MUST NOT BE DEFINED HERE)
//  - Wine/search domain models (WineCard, SearchResponse, SearchMeta) -> defined in Models.swift
//  - SSE transport wrappers (APIStreamEvent) -> defined in APIStreamEvent.swift
//  - Networking (APIClient)
//
//  RULES
//  - This file may *reference* WineCard as payload, but must never redefine it.
//  - Keep chat domain independent from transport/parsing concerns.
//
//

import Foundation

    enum ChatDomain {

        struct Message: Identifiable, Codable {
            enum Role: String, Codable { case user, assistant }

            let id: UUID
            let role: Role
            var text: String
            var wines: [WineCard]?

            init(id: UUID = UUID(), role: Role, text: String, wines: [WineCard]? = nil) {
                self.id = id
                self.role = role
                self.text = text
                self.wines = wines
            }
        }

        enum SortMode: String, CaseIterable, Identifiable, Codable {
            case relevance

            // PREMIUM modes
            case quality
            case price_value
            case match

            // existing modes
            case price_asc
            case price_desc
            case rating
            case popular

            var id: String { rawValue }

            // Backend mapping (coerente con backend attuale)
            var apiValue: String {
                switch self {
                case .quality:
                    return "quality"
                case .price_value:
                    return "price_value"
                case .match:
                    return "match"
                default:
                    return rawValue
                }
            }

            var label: String {
                switch self {
                case .relevance: return "Rilevanza"

                case .quality: return "🔬 Qualità"
                case .price_value: return "💰 Prezzo/Valore"
                case .match: return "🎯 Match"

                case .price_asc: return "Prezzo ↑"
                case .price_desc: return "Prezzo ↓"
                case .rating: return "Rating"
                case .popular: return "Popolari"
                }
            }

            /// Comparator locale per ordinare anche nei Preferiti.
            /// Nota: la sorgente di verità per l’ordinamento in Preferiti è l’estensione in Models.swift
            /// (Array where Element == WineCard). Questo less() resta utile come helper generico.
            func less(_ a: WineCard, _ b: WineCard) -> Bool {
                switch self {

                case .price_asc:
                    return (a.price ?? .infinity) < (b.price ?? .infinity)

                case .price_desc:
                    return (a.price ?? -1) > (b.price ?? -1)

                case .quality:
                    let qa = a.__quality_score ?? a.rating_overall ?? 0
                    let qb = b.__quality_score ?? b.rating_overall ?? 0
                    if qa != qb { return qa > qb }
                    return a.name.localizedCaseInsensitiveCompare(b.name) == .orderedAscending

                case .price_value:
                    let va = a.__value_score ?? 0
                    let vb = b.__value_score ?? 0
                    if va != vb { return va > vb }

                    let pa = a.price ?? .infinity
                    let pb = b.price ?? .infinity
                    if pa != pb { return pa < pb }

                    return a.name.localizedCaseInsensitiveCompare(b.name) == .orderedAscending

                case .match:
                    let ma = a.__match_score ?? 0
                    let mb = b.__match_score ?? 0
                    if ma != mb { return ma > mb }

                    let sa = a.score ?? 0
                    let sb = b.score ?? 0
                    if sa != sb { return sa > sb }

                    return a.name.localizedCaseInsensitiveCompare(b.name) == .orderedAscending

                case .rating:
                    let ra = a.rating_overall ?? 0
                    let rb = b.rating_overall ?? 0
                    if ra != rb { return ra > rb }
                    return a.name.localizedCaseInsensitiveCompare(b.name) == .orderedAscending

                case .popular:
                    let pa = a.popularity ?? 0
                    let pb = b.popularity ?? 0
                    if pa != pb { return pa > pb }
                    return a.name.localizedCaseInsensitiveCompare(b.name) == .orderedAscending

                case .relevance:
                    let sa = a.score ?? 0
                    let sb = b.score ?? 0
                    if sa != sb { return sa > sb }
                    return a.name.localizedCaseInsensitiveCompare(b.name) == .orderedAscending
                }
            }
        }
    }

    /*
     Nota rapida (non cambio ora)
     - Se vuoi eliminare la duplicazione tra SortMode.less() e Models.swift sorted(by:),
       si può fare in uno step dedicato (refactor), ma NON lo faccio ora per evitare regressioni.
    */
