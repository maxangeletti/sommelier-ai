//
//  ChatViewModel.swift
//  SommelierAI
//
//  ENGINE: SommelierAI iOS
//  VERSION: v0.9  (POST-FREEZE)
//  BUILD: SAFE + PERF + A9 + STREAM-LIMIT + PRICE-GUARD
//
//  SCOPE (QUESTO FILE):
//  - Stream SSE live + commit
//  - Tier gating sort
//  - Persistenza sort
//  - Post-process prezzo (bucket + refill)  ✅ (modificato: NO bucket-match su price sort)
//  - Local price guard (NO regressioni backend) ✅ (attualmente disabilitato)
//  - Local color guard ✅
//  - Stream limit override (fix default=3)
//  - UI banner su sort non disponibile (popular/rating) ✅
//  - UI-only: filtro vitigno + filtro prezzo max (client-side) ✅
//
//  REGOLE:
//  - ZERO refactor
//  - ZERO regressioni
//  - PATCH MINIME
//  - Compatibilità backend FREEZE
//

import Foundation
import SwiftUI
import Combine

enum RankingMode: String, CaseIterable, Identifiable {
    case standard = "relevance"
    case smart = "relevance_a9v2"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .standard: return "Standard"
        case .smart: return "Smart"
        }
    }
}

@MainActor
final class ChatViewModel: ObservableObject {

    // T0/T1: Tier store injected from View (optional)
    weak var tierStore: TierStore?

    @Published var messages: [ChatDomain.Message] = []
    @Published var isLoading: Bool = false
    @Published var streamTick: Int = 0
    @Published var didStartNewMessageTick: Int = 0
    @Published var suggestions: [String] = []

    // ✅ Banner temporaneo quando forziamo sort
    @Published var forcedSortMessage: String? = nil

    // ✅ No didSet: gating + save + rerun gestiti in un solo punto (sink)
    @Published var selectedSort: ChatDomain.SortMode = .relevance

    @Published var rerunScrollTick: Int = 0
    @Published var rerunScrollTargetID: String? = nil

    // ✅ Smart Ranking toggle (A9v2) — usato SOLO quando selectedSort == .relevance
    @Published var rankingMode: RankingMode = .standard

    // ✅ UI-only: filtro vitigno (client-side)
    @Published var selectedGrapeFilter: String = "Tutti"
    @Published var grapeFilterOptions: [String] = ["Tutti"]

    @Published var selectedColorFilter: String = "Tutti"      // red/white/rose/orange
    @Published var selectedIntensityFilter: String = "Tutti"  // low/medium/high

    // ✅ UI-only: filtro prezzo max (client-side)
    @Published var maxPriceFilter: Double = 0   // 0 = disattivo

    private let api = APIClient()
    private let storageKey = "sommelier_chat_v1"
    private let sortKey = "sommelier_sort_chat"
    private let rankingKey = "sommelier_ranking_mode_chat"
    private let grapeKey = "sommelier_grape_filter_chat"

    private var lastQuery: String?
    private var lastAssistantIndex: Int?
    private var currentStreamTask: Task<Void, Never>?

    @Published private var draftText: String = ""
    private var cancellables = Set<AnyCancellable>()

    // ✅ evita loop quando forziamo selectedSort a .relevance
    private var isAdjustingSort: Bool = false

    // ✅ SORGENTE "BASE" per filtri UI-only (vitigno/prezzo).
    // Contiene SEMPRE l'ultimo set completo (dopo color guard) dell'ultimo messaggio assistente.
    private var lastAssistantBaseWines: [WineCard] = []

    init() {
        load()

        if let raw = UserDefaults.standard.string(forKey: sortKey),
           let s = ChatDomain.SortMode(rawValue: raw) {
            selectedSort = s
        }

        if let raw = UserDefaults.standard.string(forKey: rankingKey),
           let m = RankingMode(rawValue: raw) {
            rankingMode = m
        }

        if let raw = UserDefaults.standard.string(forKey: grapeKey),
           !raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            selectedGrapeFilter = raw
        }

        Task {
            if let s = try? await api.suggestions() {
                self.suggestions = s
            } else {
                self.suggestions = []
            }
        }

        $draftText
            .removeDuplicates()
            .debounce(for: .milliseconds(350), scheduler: RunLoop.main)
            .sink { _ in }
            .store(in: &cancellables)

        // ✅ sort: gating + persistenza + rerun
        $selectedSort
            .dropFirst()
            .sink { [weak self] newSort in
                guard let self = self else { return }
                guard !self.isAdjustingSort else { return }

                // ✅ Dataset gating: popular/rating non affidabili → forza relevance + banner
                if newSort == .popular || newSort == .rating {
                    self.isAdjustingSort = true
                    self.selectedSort = .relevance
                    self.isAdjustingSort = false

                    self.forcedSortMessage = "Ordinamento non disponibile con questo dataset. Uso Rilevanza."
                    DispatchQueue.main.asyncAfter(deadline: .now() + 4.0) {
                        self.forcedSortMessage = nil
                    }
                    return
                }

                // 🔒 Tier gating (Free → solo A2/Relevance)
                if self.tierStore?.tier == .free && newSort != .relevance {
                    self.isAdjustingSort = true
                    self.selectedSort = .relevance
                    self.isAdjustingSort = false
                    return
                }

                UserDefaults.standard.set(self.selectedSort.rawValue, forKey: self.sortKey)
                self.rerunLastQuery()
            }
            .store(in: &cancellables)

        // ✅ rankingMode: persist + rerun
        $rankingMode
            .dropFirst()
            .sink { [weak self] _ in
                guard let self = self else { return }
                UserDefaults.standard.set(self.rankingMode.rawValue, forKey: self.rankingKey)
                self.rerunLastQuery()
            }
            .store(in: &cancellables)

        // ✅ filtro vitigno: ricalcola localmente dall'ultimo "base"
        $selectedGrapeFilter
            .dropFirst()
            .sink { [weak self] _ in
                guard let self = self else { return }
                UserDefaults.standard.set(self.selectedGrapeFilter, forKey: self.grapeKey) // ✅ persist
                self.reapplyLocalFiltersToLastAssistant()
            }
            .store(in: &cancellables)

        // ✅ filtro prezzo max: ricalcola localmente dall'ultimo "base"
        $maxPriceFilter
            .dropFirst()
            .sink { [weak self] _ in
                guard let self = self else { return }
                self.reapplyLocalFiltersToLastAssistant()
            }
            .store(in: &cancellables)
        // ✅ filtro colore: ricalcola localmente dall'ultimo "base"
        $selectedColorFilter
            .dropFirst()
            .sink { [weak self] _ in
                guard let self = self else { return }
                self.reapplyLocalFiltersToLastAssistant()
            }
            .store(in: &cancellables)

        // ✅ filtro intensità: ricalcola localmente dall'ultimo "base"
        $selectedIntensityFilter
            .dropFirst()
            .sink { [weak self] _ in
                guard let self = self else { return }
                self.reapplyLocalFiltersToLastAssistant()
            }
            .store(in: &cancellables)
    }

    // ✅ chiamalo dalla View quando inietti TierStore
    func attachTierStore(_ store: TierStore?) {
        self.tierStore = store
        enforceTierOnCurrentSortIfNeeded()
    }

    private func enforceTierOnCurrentSortIfNeeded() {
        guard !isAdjustingSort else { return }
        guard tierStore?.tier == .free else { return }
        guard selectedSort != .relevance else { return }

        isAdjustingSort = true
        selectedSort = .relevance
        isAdjustingSort = false
        UserDefaults.standard.set(ChatDomain.SortMode.relevance.rawValue, forKey: sortKey)
    }

    // ✅ AIRBAG: normalizza sort prima di una request.
    private func normalizedSortForRequest() -> ChatDomain.SortMode {
        let isFreeOrUnknown: Bool
        if let t = tierStore?.tier {
            isFreeOrUnknown = (t == .free)
        } else {
            isFreeOrUnknown = true
        }

        if isFreeOrUnknown {
            if selectedSort != .relevance {
                isAdjustingSort = true
                selectedSort = .relevance
                isAdjustingSort = false
                UserDefaults.standard.set(ChatDomain.SortMode.relevance.rawValue, forKey: sortKey)
            }
            return .relevance
        }
        return selectedSort
    }

    private func backendSortForSelectedSort(_ displaySort: ChatDomain.SortMode) -> ChatDomain.SortMode {
        displaySort
    }

    private func backendSortString(displaySort: ChatDomain.SortMode) -> String {
        if displaySort == .relevance {
            return rankingMode.rawValue
        }
        return backendSortForSelectedSort(displaySort).apiValue
    }

    // ✅ Post-process (PATCH):
    // - Applica sempre filtro prezzo max (se attivo)
    // - price_asc/price_desc: dedup + sort locale per prezzo
    private func postProcessWinesForDisplay(_ wines: [WineCard], displaySort: ChatDomain.SortMode) -> [WineCard] {

        // ✅ UI-only: prezzo max (0 = off)
        let priceFiltered: [WineCard] = {
            guard maxPriceFilter > 0 else { return wines }
            return wines.filter { w in
                guard let p = w.price, p.isFinite else { return false }
                return p <= maxPriceFilter
            }
        }()

        switch displaySort {
        case .price_asc, .price_desc:
            var seen = Set<String>()
            let unique = priceFiltered.filter { seen.insert($0.id).inserted }
            return unique.sorted { displaySort.less($0, $1) }

        default:
            return priceFiltered
        }
    }

    // ✅ Guard locale prezzo: DISABILITATO (backend è source of truth)
    private func passesLocalPriceGuard(_ wine: WineCard, query: String) -> Bool { true }

    private func expectedColorFromQuery(_ query: String) -> String? {
        let q = query.lowercased()
        if q.contains("rosso") || q.contains("rossi") || q.contains("red") { return "red" }
        if q.contains("bianco") || q.contains("bianchi") || q.contains("white") { return "white" }
        if q.contains("rosé") || q.contains("rose") || q.contains("rosato") { return "rose" }
        return nil
    }

    // MARK: - Extra UI filters (color + intensity)

    private func normalizeColor(_ s: String?) -> String? {
        guard let s = s?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(),
              !s.isEmpty else { return nil }
        if ["red","white","rose","orange"].contains(s) { return s }
        return nil
    }

    private func normalizeIntensity(_ s: String?) -> String? {
        guard let s = s?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(),
              !s.isEmpty else { return nil }
        if ["low","medium","high"].contains(s) { return s }
        return nil
    }

    private func passesLocalColorGuard(_ wine: WineCard, query: String) -> Bool {
        guard let expected = expectedColorFromQuery(query) else { return true }

        if expected == "red" {
            let n = wine.name.lowercased()
            let d = (wine.denomination ?? "").lowercased()
            if n.contains("franciacorta") || d.contains("franciacorta") {
                return false
            }
        }

        if let c = wine.color?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(),
           !c.isEmpty {
            return c == expected || (expected == "red" && (c == "rosso" || c == "rouge")) || (expected == "white" && (c == "bianco" || c == "blanc")) || (expected == "rose" && (c == "rosato"))
        }

        if let detail = wine.color_detail?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(),
           !detail.isEmpty {
            switch expected {
            case "red":
                return detail.contains("ross") || detail.contains("rubin") || detail.contains("granat") || detail.contains("porpora") || detail.contains("red") || detail.contains("ruby")
            case "white":
                return detail.contains("bianc") || detail.contains("giall") || detail.contains("paglier") || detail.contains("straw")
            case "rose":
                return detail.contains("ros") || detail.contains("cerasuol") || detail.contains("salmon")
            default:
                return true
            }
        }

        return true
    }

    // MARK: - Grape filter (UI-only)

    private func normalizeGrape(_ s: String) -> String {
        s.trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "'", with: "'")
    }

    private func extractGrapes(from raw: String?) -> [String] {
        guard let raw, !raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return [] }

        let unified = raw
            .replacingOccurrences(of: ",", with: "|")
            .replacingOccurrences(of: ";", with: "|")
            .replacingOccurrences(of: "/", with: "|")
            .replacingOccurrences(of: "•", with: "|")

        let parts = unified
            .split(separator: "|")
            .map { normalizeGrape(String($0)) }
            .filter { !$0.isEmpty }

        var seen = Set<String>()
        return parts.filter { seen.insert($0.lowercased()).inserted }
    }

    private func applyGrapeFilter(_ wines: [WineCard]) -> [WineCard] {
        let sel = selectedGrapeFilter.trimmingCharacters(in: .whitespacesAndNewlines)
        guard sel != "Tutti" else { return wines }

        let target = sel.lowercased()
        return wines.filter { w in
            extractGrapes(from: w.grapes).contains { $0.lowercased() == target }
        }
    }

    private func applyAllLocalFilters(_ wines: [WineCard]) -> [WineCard] {

        // 1) vitigno
        var out = applyGrapeFilter(wines)

        // 2) colore
        if selectedColorFilter != "Tutti" {
            let target = selectedColorFilter.lowercased()
            out = out.filter { normalizeColor($0.color) == target }
        }

        // 3) intensità
        if selectedIntensityFilter != "Tutti" {
            let target = selectedIntensityFilter.lowercased()
            out = out.filter { normalizeIntensity($0.intensity) == target }
        }

        // 4) prezzo (già gestito nel postProcess)
        out = postProcessWinesForDisplay(out, displaySort: selectedSort)

        return out
    }

    private func updateGrapeOptions(from wines: [WineCard]) {
        var set = Set<String>()
        for w in wines {
            for g in extractGrapes(from: w.grapes) {
                set.insert(g)
            }
        }

        let sorted = set
            .filter { $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() != "tutti" }
            .sorted { $0.localizedCaseInsensitiveCompare($1) == .orderedAscending }

        let options = ["Tutti"] + sorted
        grapeFilterOptions = options

        if !options.contains(selectedGrapeFilter) {
            selectedGrapeFilter = "Tutti"
        }
    }

    // ✅ Applica vitigno + prezzo max partendo SEMPRE dalla sorgente base
    private func reapplyLocalFiltersToLastAssistant() {

        // safety assoluta contro race / index invalid
        guard !messages.isEmpty else { return }
        guard let idx = lastAssistantIndex else { return }
        guard idx >= 0 && idx < messages.count else { return }

        // se base vuoto → non fare nulla (evita crash e filtri su nil)
        guard !lastAssistantBaseWines.isEmpty else { return }

        updateGrapeOptions(from: lastAssistantBaseWines)

        let grapeFiltered = applyGrapeFilter(lastAssistantBaseWines)
        let processed = postProcessWinesForDisplay(grapeFiltered, displaySort: selectedSort)

        // safety write
        if idx < messages.count {
            messages[idx].wines = processed
        }

        streamTick += 1
        save()
    }

    // ✅ NEW (chirurgico): prepara opzioni per sheet prima di aprirla
    func prepareGrapeSheetOptions() {
        // 1) se abbiamo base, usala
        if !lastAssistantBaseWines.isEmpty {
            updateGrapeOptions(from: lastAssistantBaseWines)
            return
        }

        // 2) fallback robusto: cerca l'ULTIMO messaggio assistente con wines non vuoti
        if let msg = messages.reversed().first(where: { $0.role == .assistant && (($0.wines?.isEmpty) == false) }),
           let wines = msg.wines,
           !wines.isEmpty {
            updateGrapeOptions(from: wines)
            return
        }

        // 3) fallback duro
        grapeFilterOptions = ["Tutti"]
        if selectedGrapeFilter.isEmpty { selectedGrapeFilter = "Tutti" }
    }

    func updateDraft(_ text: String) { draftText = text }

    func clear() {
        currentStreamTask?.cancel()
        messages.removeAll()
        lastQuery = nil
        lastAssistantIndex = nil
        lastAssistantBaseWines = []
        grapeFilterOptions = ["Tutti"]
        selectedGrapeFilter = "Tutti"
        save()
    }

    func send(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        currentStreamTask?.cancel()
        lastQuery = trimmed

        // ✅ Reset automatico filtro vitigno su nuova query
        if selectedGrapeFilter != "Tutti" {
            selectedGrapeFilter = "Tutti"
            UserDefaults.standard.set("Tutti", forKey: grapeKey)
        }

        messages.append(ChatDomain.Message(role: .user, text: trimmed))
        messages.append(ChatDomain.Message(role: .assistant, text: "", wines: []))
        lastAssistantIndex = messages.count - 1

        // reset base wines for this new answer
        lastAssistantBaseWines = []

        save()

        didStartNewMessageTick += 1
        runStreamLive(query: trimmed, assistantIndex: lastAssistantIndex)
    }

    private func rerunLastQuery() {
        guard let q = lastQuery,
              let idx = lastAssistantIndex,
              messages.indices.contains(idx) else { return }

        currentStreamTask?.cancel()
        runStreamCommitOnFinal(query: q, assistantIndex: idx)
    }

    private func runStreamLive(query: String, assistantIndex: Int?) {
        guard let idx = assistantIndex, messages.indices.contains(idx) else { return }

        let displaySort = normalizedSortForRequest()
        var backendSort = backendSortString(displaySort: displaySort)

        if displaySort == .popular || displaySort == .rating {
            backendSort = ChatDomain.SortMode.relevance.apiValue
        }

        let limit = 20
        isLoading = true

        currentStreamTask = Task {
            defer { isLoading = false; save() }

            do {
                var deltaCounter = 0
                var gotAnyWine = false
                messages[idx].wines = []

                for try await ev in api.searchStream(query: query, sortMode: backendSort, limit: limit) {
                    if Task.isCancelled { return }

                    let type = ev.type.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()

                    if type == "delta" {
                        if let t = ev.text, !t.isEmpty {
                            messages[idx].text += t
                            deltaCounter += 1
                            if deltaCounter % 2 == 0 { streamTick += 1 }
                        }

                        if let w = ev.wine,
                           passesLocalPriceGuard(w, query: query),
                           passesLocalColorGuard(w, query: query) {

                            gotAnyWine = true

                            // aggiorna BASE
                            lastAssistantBaseWines.append(w)
                            updateGrapeOptions(from: lastAssistantBaseWines)

                            // ✅ PATCH (Step 5): applica MULTI-FILTRO LIVE (vitigno+prezzo+colore+intensità)
                            let processed = applyAllLocalFilters(lastAssistantBaseWines)
                            messages[idx].wines = processed

                            deltaCounter += 1
                            if deltaCounter % 2 == 0 { streamTick += 1 }
                        }

                    } else if type == "final" {
                        if let m = ev.message, !m.isEmpty {
                            messages[idx].text = m
                        } else if messages[idx].text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                            messages[idx].text = gotAnyWine ? "Ecco i migliori risultati:" : "Nessun vino trovato per questa ricerca. Prova con termini diversi."
                        }

                        // ✅ base = risultati completi (dopo color guard)
                        let base: [WineCard] = {
                            if let finalResults = ev.results {
                                return finalResults.filter { passesLocalColorGuard($0, query: query) }
                            }
                            if let finalWines = ev.wines {
                                return finalWines.filter { passesLocalColorGuard($0, query: query) }
                            }
                            return (messages[idx].wines ?? []).filter { passesLocalColorGuard($0, query: query) }
                        }()

                        // ✅ salva sorgente base + aggiorna opzioni vitigno
                        self.lastAssistantBaseWines = base
                        self.updateGrapeOptions(from: base)

                        // ✅ PATCH (Step 4): applica MULTI-FILTRO FINAL (vitigno+prezzo+colore+intensità)
                        let allProcessed = self.applyAllLocalFilters(base)
                        
                        // ✅ PAGINAZIONE: Mostra inizialmente solo i primi 10 vini
                        let initialLimit = 10
                        let processed = Array(allProcessed.prefix(initialLimit))
                        
                        // ✅ PAGINAZIONE: salva totalCount e currentLimit
                        let totalCount = ev.meta?.total_count ?? base.count
                        let currentLimit = min(initialLimit, base.count)  // 🔧 FIX: inizia con 10 vini
                        
                        withAnimation(.easeInOut(duration: 0.18)) {
                            messages[idx].wines = processed
                            messages[idx].totalCount = totalCount
                            messages[idx].currentLimit = currentLimit
                        }

                        streamTick += 1
                        save()
                        return
                    }
                }

            } catch {
                if error is CancellationError { return }
                messages[idx].text = "Errore: non riesco a contattare il server."
                messages[idx].wines = nil
                streamTick += 1
                save()
            }
        }
    }

    private func runStreamCommitOnFinal(query: String, assistantIndex: Int) {
        let idx = assistantIndex

        let displaySort = normalizedSortForRequest()
        var backendSort = backendSortString(displaySort: displaySort)

        if displaySort == .popular || displaySort == .rating {
            backendSort = ChatDomain.SortMode.relevance.apiValue
        }

        let limit = 20
        isLoading = true

        currentStreamTask = Task {
            defer { isLoading = false; save() }

            do {
                var bufferedText = ""
                var bufferedWines: [WineCard] = []
                var gotAnyWine = false

                for try await ev in api.searchStream(query: query, sortMode: backendSort, limit: limit) {
                    if Task.isCancelled { return }

                    let type = ev.type.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()

                    if type == "delta" {
                        if let t = ev.text, !t.isEmpty { bufferedText += t }

                        if let w = ev.wine,
                           passesLocalPriceGuard(w, query: query),
                           passesLocalColorGuard(w, query: query) {
                            bufferedWines.append(w)
                            gotAnyWine = true
                        }

                    } else if type == "final" {
                        if let m = ev.message, !m.isEmpty {
                            messages[idx].text = m
                        } else {
                            let bt = bufferedText.trimmingCharacters(in: .whitespacesAndNewlines)
                            messages[idx].text = bt.isEmpty
                                ? (gotAnyWine ? "Ecco i migliori risultati:" : "Ecco i risultati:")
                                : bt
                        }

                        let base: [WineCard] = {
                            if let finalResults = ev.results {
                                return finalResults.filter { passesLocalColorGuard($0, query: query) }
                            }
                            if let finalWines = ev.wines {
                                return finalWines.filter { passesLocalColorGuard($0, query: query) }
                            }
                            return bufferedWines.filter { passesLocalColorGuard($0, query: query) }
                        }()

                        self.lastAssistantBaseWines = base
                        self.updateGrapeOptions(from: base)

                        // ✅ PATCH (Step 4/5): applica MULTI-FILTRO anche nel commit (no streaming)
                        let allProcessed = self.applyAllLocalFilters(base)
                        
                        // ✅ PAGINAZIONE: Mostra inizialmente solo i primi 10 vini
                        let initialLimit = 10
                        let processed = Array(allProcessed.prefix(initialLimit))
                        
                        // ✅ PAGINAZIONE: salva totalCount e currentLimit
                        let totalCount = ev.meta?.total_count ?? base.count
                        let currentLimit = min(initialLimit, base.count)  // 🔧 FIX: inizia con 10 vini
                        
                        withAnimation(.easeInOut(duration: 0.18)) {
                            messages[idx].wines = processed
                            messages[idx].totalCount = totalCount
                            messages[idx].currentLimit = currentLimit
                        }

                        // ✅ usa lo STESSO schema anchor della UI: wine:<msgUUID>:0
                        rerunScrollTargetID = "wine:\(messages[idx].id.uuidString):0"
                        rerunScrollTick += 1

                        streamTick += 1
                        save()
                        return
                    }
                }

            } catch {
                if error is CancellationError { return }
            }
        }
    }

    private func wineAnchorID(messageID: UUID, wineID: String) -> String {
        "wine:\(messageID.uuidString):\(wineID)"
    }

    private func save() {
        if let data = try? JSONEncoder().encode(messages) {
            UserDefaults.standard.set(data, forKey: storageKey)
        }
    }

    private func load() {
        guard let data = UserDefaults.standard.data(forKey: storageKey),
              let arr = try? JSONDecoder().decode([ChatDomain.Message].self, from: data) else { return }
        messages = arr

        lastAssistantIndex = messages.lastIndex(where: { $0.role == .assistant })
        if let lastUser = messages.last(where: { $0.role == .user }) {
            lastQuery = lastUser.text
        }

        // ✅ NEW (chirurgico): ricostruisci base+options dopo restart
        if let idx = lastAssistantIndex,
           messages.indices.contains(idx),
           let wines = messages[idx].wines,
           !wines.isEmpty {
            lastAssistantBaseWines = wines
            updateGrapeOptions(from: wines)
        }
    }
    
    // MARK: - Paginazione
    
    /// Carica altri 5 vini per il messaggio specificato
    func loadMore(for messageId: UUID) {
        guard let msgIndex = messages.firstIndex(where: { $0.id == messageId }),
              messages.indices.contains(msgIndex),
              let totalCount = messages[msgIndex].totalCount,
              let currentLimit = messages[msgIndex].currentLimit,
              currentLimit < totalCount,
              currentLimit < 20 else { // Max 20 vini totali
            return
        }
        
        let newLimit = min(currentLimit + 5, 20, totalCount)
        
        // 🔧 FIX: Prendi i primi newLimit vini BASE, poi applica filtri
        let baseWines = lastAssistantBaseWines
        let baseSlice = Array(baseWines.prefix(newLimit))
        let processed = applyAllLocalFilters(baseSlice)
        
        withAnimation(.easeInOut(duration: 0.25)) {
            messages[msgIndex].wines = processed
            messages[msgIndex].currentLimit = newLimit
        }
        
        streamTick += 1
        save()
    }
}

/*
 Nota rapida (non cambio ora)
 - FIX sheet vitigno vuota: aggiunta prepareGrapeSheetOptions() + rebuild opzioni in load().
 - Multi-filtro: applyAllLocalFilters applicato in LIVE + FINAL + COMMIT (patch minime).
 - Nessun refactor, solo patch minime.
*/
