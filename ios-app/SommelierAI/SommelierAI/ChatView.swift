//
//  ChatView.swift
//  SommelierAI
//
//  PURPOSE
//  Main chat screen UI:
//  - Renders chat messages (user/assistant)
//  - Renders wine result cards (ranking + highlights)
//  - Hosts filter chips (grape/color/intensity) + price slider
//  - Handles scroll behavior (top/bottom anchors, rerun scroll)
//
//  SCOPE (THIS FILE)
//  - UI layout + view state (SwiftUI)
//  - Scroll anchoring + compact filter bar behavior
//  - Sheets for UI-only filters
//  - Visual polish (badges, stars, micro-reason #1)
//
//  OUT OF SCOPE (MUST NOT BE DEFINED HERE)
//  - Networking (APIClient.swift)
//  - Streaming event model (APIStreamEvent.swift)
//  - Domain models (Models.swift)
//  - Request/ranking logic (ChatViewModel.swift)
//
//  RULES
//  - Patch only (no refactor)
//  - Do not touch Models unless strictly necessary
//  - Keep changes localized and UI-only when possible
//

import SwiftUI

private let rankingDebugMode = false

struct ChatView: View {
    @StateObject private var vm = ChatViewModel()
    @EnvironmentObject private var favoritesStore: FavoritesStore
    @EnvironmentObject private var tierStore: TierStore

    @State private var inputText: String = ""
    @FocusState private var isInputFocused: Bool

    private let bottomID = "bottom"

    @State private var isNearBottom: Bool = true
    @State private var viewportHeight: CGFloat = 0
    @State private var userIsInteracting: Bool = false

    @State private var expandedSuggestion: String? = nil
    @State private var suggestionsSeed: Int = 0
    @State private var forceScrollToBottomTick: Int = 0

    // ✅ Sheets filtri
    @State private var showGrapeSheet: Bool = false
    @State private var showColorSheet: Bool = false
    @State private var showIntensitySheet: Bool = false

    // ✅ Compact filter bar (Apple Music-ish)
    @State private var scrollOffsetY: CGFloat = 0
    @State private var isFilterBarCompact: Bool = false

    // ✅ UI toggles (manual)
    @State private var showSuggestions: Bool = false
    @State private var showPriceBar: Bool = false

    @State private var lastResultMessageUUID: String? = nil
    @State private var lastResultHadWines: Bool = false
    @State private var expandedWineIDs: Set<String> = []

    private var allowedChatSortModes: [ChatDomain.SortMode] {
        tierStore.tier == .free ? [.relevance] : ChatDomain.SortMode.allCases
    }

    private var allowedRankingModes: [RankingMode] {
        tierStore.tier == .free ? [.standard, .smart] : RankingMode.allCases
    }

    // ✅ FIX (chirurgico): niente nesting, icona sempre visibile
    private var grapeSymbolName: String { "leaf" }

    // ✅ Count vini attualmente mostrati nell’ultimo messaggio assistente (post-filtri)
    private var displayedWineCount: Int {
        guard let msg = vm.messages.last(where: { $0.role == .assistant }),
              let wines = msg.wines else { return 0 }
        return wines.count
    }

    // ✅ Colore badge dinamico in base al numero vini
    private var grapeBadgeColor: Color {
        let c = displayedWineCount
        if c <= 3 { return .green }      // pochi
        if c <= 10 { return .orange }    // medio
        return .red                      // molti
    }

    // ✅ Titoli bottoni filtri
    private var grapeButtonTitle: String {
        let sel = vm.selectedGrapeFilter.trimmingCharacters(in: .whitespacesAndNewlines)
        if sel.lowercased() == "tutti" || sel.isEmpty { return "Vitigno" }
        return sel
    }

    private var colorButtonTitle: String {
        let sel = vm.selectedColorFilter.trimmingCharacters(in: .whitespacesAndNewlines)
        if sel.lowercased() == "tutti" || sel.isEmpty { return "Colore" }
        return sel.capitalized
    }

    private var intensityButtonTitle: String {
        let sel = vm.selectedIntensityFilter.trimmingCharacters(in: .whitespacesAndNewlines)
        if sel.lowercased() == "tutti" || sel.isEmpty { return "Intensità" }
        return sel.capitalized
    }

    var body: some View {
        VStack(spacing: 0) {

            // ✅ Banner (UI-only) — quando il VM forza il sort (es: Popular/Rating → Relevance)
            if let msg = vm.forcedSortMessage {
                Text(msg)
                    .font(.caption)
                    .padding(8)
                    .frame(maxWidth: .infinity)
                    .background(Color.yellow.opacity(0.2))
            }

            // ✅ Barra filtri SCROLLABILE - visibile solo dopo primo risultato
            if lastResultHadWines {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 12) {

                        Button {
                            vm.prepareGrapeSheetOptions()
                            showGrapeSheet = true
                        } label: {
                            filterChip(
                                system: grapeSymbolName,
                                title: grapeButtonTitle,
                                badge: displayedWineCount > 0 ? "\(displayedWineCount)" : nil,
                                badgeColor: grapeBadgeColor
                            )
                        }

                        Button { showColorSheet = true } label: {
                            filterChip(system: "paintpalette", title: colorButtonTitle)
                        }

                        Button { showIntensitySheet = true } label: {
                            filterChip(system: "flame", title: intensityButtonTitle)
                        }
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, isFilterBarCompact ? 4 : 8)
                }
                .scaleEffect(isFilterBarCompact ? 0.92 : 1.0, anchor: .top)
                .opacity(isFilterBarCompact ? 0.94 : 1.0)
                .background(AppColors.backgroundPrimary)

                Divider()
            }

            let cleanSuggestions = vm.suggestions
                .map { sanitizeSuggestion($0) }
                .filter { !$0.isEmpty }

            // ✅ Suggerimenti: apri/chiudi su tap (non più legati allo scroll)
            if !cleanSuggestions.isEmpty {
                Button {
                    withAnimation(.easeInOut(duration: 0.15)) {
                        showSuggestions.toggle()
                        expandedSuggestion = nil
                    }
                } label: {
                    HStack(spacing: 8) {
                        Text("Prova a chiedere a Sommy:")
                            .font(.footnote)
                            .foregroundStyle(.secondary)

                        Spacer()

                        Image(systemName: showSuggestions ? "chevron.up" : "chevron.down")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 12)
                    .padding(.top, 10)
                    .padding(.bottom, showSuggestions ? 6 : 10)
                }
                .buttonStyle(.plain)

                if showSuggestions {
                    let pool = cleanSuggestions
                    let picked: [String] = {
                        guard !pool.isEmpty else { return [] }
                        var rng = SeededGenerator(seed: UInt64(suggestionsSeed))
                        let shuffled = pool.shuffled(using: &rng)
                        return Array(shuffled.prefix(3))
                    }()

                    LazyVGrid(
                        columns: [
                            GridItem(.flexible(), spacing: 8),
                            GridItem(.flexible(), spacing: 8)
                        ],
                        spacing: 8
                    ) {
                        ForEach(picked, id: \.self) { s in
                            let isExpanded = (expandedSuggestion == s)

                            Button {
                                if expandedSuggestion != s {
                                    withAnimation(.easeInOut(duration: 0.15)) {
                                        expandedSuggestion = s
                                    }
                                } else {
                                    vm.send(s)
                                    forceScrollToBottomTick += 1
                                    inputText = ""
                                    withAnimation(.easeInOut(duration: 0.15)) {
                                        expandedSuggestion = nil
                                        showSuggestions = false
                                    }
                                }
                            } label: {
                                Text(s)
                                    .font(.footnote)
                                    .foregroundColor(.primary)
                                    .lineLimit(isExpanded ? 3 : 1)
                                    .multilineTextAlignment(.center)
                                    .frame(maxWidth: .infinity, alignment: .center)
                                    .padding(.vertical, 8)
                                    .padding(.horizontal, 10)
                                    .background(.thinMaterial)
                                    .clipShape(Capsule())
                            }
                            .buttonStyle(.plain)
                            .simultaneousGesture(
                                LongPressGesture(minimumDuration: 0.35).onEnded { _ in
                                    vm.send(s)
                                    forceScrollToBottomTick += 1
                                    inputText = ""
                                    withAnimation(.easeInOut(duration: 0.15)) {
                                        expandedSuggestion = nil
                                        showSuggestions = false
                                    }
                                }
                            )
                        }

                        Button {
                            withAnimation(.easeInOut(duration: 0.15)) {
                                suggestionsSeed += 1
                                expandedSuggestion = nil
                            }
                        } label: {
                            Text("Altri")
                                .font(.footnote)
                                .foregroundColor(.primary)
                                .lineLimit(1)
                                .frame(maxWidth: .infinity, alignment: .center)
                                .padding(.vertical, 8)
                                .padding(.horizontal, 10)
                                .background(.thinMaterial)
                                .clipShape(Capsule())
                        }
                        .buttonStyle(.plain)
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 10)

                    Divider()
                } else {
                    Divider()
                }
            }

            // ✅ Filtro Prezzo Max: tap apre/chiude; resta visibile finché maxPriceFilter > 0
            VStack(spacing: 6) {
                Button {
                    withAnimation(.easeInOut(duration: 0.15)) {
                        showPriceBar.toggle()
                    }
                } label: {
                    HStack {
                        Text("Prezzo max")
                            .font(.footnote)
                            .foregroundStyle(.secondary)

                        Spacer()

                        Text(vm.maxPriceFilter > 0 ? String(format: "€%.0f", vm.maxPriceFilter) : "OFF")
                            .font(.footnote)
                            .foregroundStyle(.secondary)

                        Image(systemName: showPriceBar ? "chevron.up" : "chevron.down")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                            .padding(.leading, 6)
                    }
                }
                .buttonStyle(.plain)

                if showPriceBar {
                    Slider(value: $vm.maxPriceFilter, in: 0...200, step: 5)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .onChange(of: vm.maxPriceFilter) { _, newValue in
                if newValue > 0 { showPriceBar = true } else { showPriceBar = false }
            }

            Divider()

            // Chat
            ScrollViewReader { proxy in
                ScrollView {
                    VStack(alignment: .leading, spacing: 12) {

                        // ✅ offset reale della scroll (serve per isFilterBarCompact)
                        Color.clear
                            .frame(height: 0)
                            .background(
                                GeometryReader { geo in
                                    Color.clear.preference(
                                        key: ScrollOffsetYKey.self,
                                        value: geo.frame(in: .named("scroll")).minY
                                    )
                                }
                            )

                        // ✅ anchor top: usato per reset scroll quando cambia ordinamento
                        Color.clear
                            .frame(height: 1)
                            .id("top")

                        ForEach(vm.messages) { msg in
                            messageRow(msg)
                        }

                        Color.clear
                            .frame(height: 1)
                            .id(bottomID)
                            .background(
                                GeometryReader { geo in
                                    Color.clear.preference(
                                        key: BottomYKey.self,
                                        value: geo.frame(in: .named("scroll")).minY
                                    )
                                }
                            )
                    }
                    .padding(12)
                }
                .coordinateSpace(name: "scroll")
                .simultaneousGesture(
                    DragGesture(minimumDistance: 2)
                        .onChanged { _ in userIsInteracting = true }
                        .onEnded { _ in
                            DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                                userIsInteracting = false
                            }
                        }
                )
                .background(
                    GeometryReader { geo in
                        Color.clear.preference(key: ViewportHeightKey.self, value: geo.size.height)
                    }
                )
                .onPreferenceChange(ViewportHeightKey.self) { h in
                    viewportHeight = h
                }
                .onPreferenceChange(ScrollOffsetYKey.self) { y in
                    scrollOffsetY = y
                    let compact = y < -28
                    if compact != isFilterBarCompact {
                        withAnimation(.easeInOut(duration: 0.18)) {
                            isFilterBarCompact = compact
                        }
                    }
                }
                .onPreferenceChange(BottomYKey.self) { bottomY in
                    let distanceToBottom = bottomY - viewportHeight
                    isNearBottom = distanceToBottom < 140
                }
                .onChange(of: vm.streamTick) { _, _ in
                    guard !userIsInteracting else { return }

                    guard let lastAssistant = vm.messages.last(where: { $0.role == .assistant }) else {
                        if isNearBottom { proxy.scrollTo(bottomID, anchor: .bottom) }
                        return
                    }

                    let msgUUID = lastAssistant.id.uuidString
                    let hasWines = (lastAssistant.wines?.isEmpty == false)

                    if hasWines {
                        if lastResultMessageUUID != msgUUID || lastResultHadWines == false {
                            lastResultMessageUUID = msgUUID
                            isInputFocused = false
                            lastResultHadWines = true

                            let firstWineAnchor = "wine:\(msgUUID):0"
                            DispatchQueue.main.async { proxy.scrollTo(firstWineAnchor, anchor: .top) }
                            DispatchQueue.main.asyncAfter(deadline: .now() + 0.12) {
                                proxy.scrollTo(firstWineAnchor, anchor: .top)
                            }
                        }
                        return
                    }

                    lastResultMessageUUID = msgUUID
                    lastResultHadWines = false

                    guard isNearBottom else { return }
                    proxy.scrollTo(bottomID, anchor: .bottom)
                }
                .onChange(of: vm.didStartNewMessageTick) { _, _ in
                    DispatchQueue.main.async { proxy.scrollTo(bottomID, anchor: .bottom) }
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.12) {
                        proxy.scrollTo(bottomID, anchor: .bottom)
                    }
                }
                .onChange(of: forceScrollToBottomTick) { _, _ in
                    DispatchQueue.main.async { proxy.scrollTo(bottomID, anchor: .bottom) }
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.12) {
                        proxy.scrollTo(bottomID, anchor: .bottom)
                    }
                }
                .onChange(of: vm.rerunScrollTick) { _, _ in
                    guard !userIsInteracting, let target = vm.rerunScrollTargetID else { return }
                    DispatchQueue.main.async { proxy.scrollTo(target, anchor: .top) }
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.12) {
                        proxy.scrollTo(target, anchor: .top)
                    }
                }
                .onChange(of: vm.selectedSort) { _, _ in
                    DispatchQueue.main.async { proxy.scrollTo("top", anchor: .top) }
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.12) {
                        proxy.scrollTo("top", anchor: .top)
                    }
                }
                .onChange(of: vm.rankingMode) { _, _ in
                    DispatchQueue.main.async { proxy.scrollTo("top", anchor: .top) }
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.12) {
                        proxy.scrollTo("top", anchor: .top)
                    }
                }
            }

            Divider()

            // Input
            HStack(spacing: 10) {
                TextField("Scrivi che vino cerchi…", text: $inputText)
                    .textFieldStyle(.roundedBorder)
                    .focused($isInputFocused)
                    .onChange(of: inputText) { _, new in
                        vm.updateDraft(new)
                    }

                Button("Invia") {
                    UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
                    vm.send(inputText)
                    forceScrollToBottomTick += 1
                    inputText = ""
                    withAnimation(.easeInOut(duration: 0.15)) {
                        expandedSuggestion = nil
                        showSuggestions = false
                    }
                }
                .tint(AppColors.accentWine)
                .disabled(vm.isLoading)
            }
            .padding(12)
        }
        .background(AppColors.backgroundPrimary)
        .onAppear { vm.attachTierStore(tierStore) }
        .navigationTitle("")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            // ✅ Cestino Reset in navbar
            ToolbarItem(placement: .topBarLeading) {
                Button { vm.clear() } label: {
                    Image(systemName: "trash")
                        .foregroundStyle(.secondary)
                }
            }

            ToolbarItem(placement: .principal) {
                Text("Sommelier AI").font(.headline)
            }

            ToolbarItem(placement: .topBarTrailing) {
                Menu {
                    Picker("Ranking", selection: $vm.rankingMode) {
                        ForEach(allowedRankingModes) { mode in
                            Text(mode.title).tag(mode)
                        }
                    }

                    Divider()

                    Picker("Ordina per", selection: $vm.selectedSort) {
                        ForEach(allowedChatSortModes) { m in
                            Text(m.label).tag(m)
                        }
                    }
                } label: {
                    Image(systemName: "arrow.up.arrow.down")
                }
            }
        }

        // ✅ Sheet vitigno — FAIL-SAFE: non passa mai lista vuota
        .sheet(isPresented: $showGrapeSheet, onDismiss: {}) {
            GrapePickerSheet(
                title: displayedWineCount > 0 ? "Vitigno (\(displayedWineCount))" : "Vitigno",
                options: vm.grapeFilterOptions.isEmpty ? ["Tutti"] : vm.grapeFilterOptions,
                selection: $vm.selectedGrapeFilter
            )
            .onAppear { vm.prepareGrapeSheetOptions() }
        }

        .sheet(isPresented: $showColorSheet) {
            SimplePickerSheet(
                title: "Colore",
                options: ["Tutti", "Red", "White", "Rose", "Orange"],
                selection: $vm.selectedColorFilter
            )
        }

        .sheet(isPresented: $showIntensitySheet) {
            SimplePickerSheet(
                title: "Intensità",
                options: ["Tutti", "Low", "Medium", "High"],
                selection: $vm.selectedIntensityFilter
            )
        }
    }

    private func sanitizeSuggestion(_ s: String) -> String {
        let trimmed = s.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "" }

        let invisible: Set<UnicodeScalar> = ["\u{200B}", "\u{200C}", "\u{200D}", "\u{FEFF}"]
        let filteredScalars = trimmed.unicodeScalars.filter { !invisible.contains($0) }
        return String(String.UnicodeScalarView(filteredScalars))
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func filterChip(
        system: String,
        title: String,
        badge: String? = nil,
        badgeColor: Color = .blue
    ) -> some View {
        HStack(spacing: 6) {
            Image(systemName: system)

            Text(title)
                .font(.subheadline)

            if let badge {
                Text(badge)
                    .font(.caption2.weight(.bold))
                    .foregroundColor(.white)
                    .padding(.horizontal, 7)
                    .padding(.vertical, 2)
                    .background(Capsule().fill(badgeColor))
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(.thinMaterial)
        .clipShape(Capsule())
    }

    // MARK: - Rows

    @ViewBuilder
    private func messageRow(_ msg: ChatDomain.Message) -> some View {
        VStack(alignment: msg.role == .user ? .trailing : .leading, spacing: 8) {
            HStack {
                if msg.role == .assistant { Spacer(minLength: 0) }

                Text(msg.text)
                    .padding(12)
                    .foregroundStyle(.primary)
                    .background(
                        msg.role == .user
                        ? AppColors.primaryWine.opacity(0.15)
                        : AppColors.backgroundSecondary
                    )
                    .clipShape(RoundedRectangle(cornerRadius: 14))

                if msg.role == .user { Spacer(minLength: 0) }
            }

            if msg.role == .assistant, let wines = msg.wines, !wines.isEmpty {
                VStack(spacing: 10) {
                    let groups = groupedWinesForDisplay(wines)

                    ForEach(groups) { g in
                        VStack(alignment: .leading, spacing: 6) {
                            wineRow(g.representative)

                            // ✅ POLISH 3 — micro reason solo per il primo
                            if g.representativeIndex == 0 {
                                Text(topReasonLabel(for: g.representative))
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }

                            if let line = vintagesLine(g.vintages) {
                                Text(line)
                                    .font(.footnote)
                                    .foregroundStyle(.secondary)
                                    .lineLimit(1)
                            }
                        }
                        .padding(10)
                        .background(
                            g.representativeIndex == 0
                            ? RoundedRectangle(cornerRadius: 14)
                                .fill(Color(red: 0.98, green: 0.94, blue: 0.78).opacity(0.6))
                            : nil
                        )
                        .overlay(
                            g.representativeIndex == 0
                            ? RoundedRectangle(cornerRadius: 14)
                                .stroke(Color.yellow.opacity(0.15), lineWidth: 1)
                            : nil
                        )
                        .id("wine:\(msg.id.uuidString):\(g.representativeIndex)")
                    }
                    
                    // ✅ PAGINAZIONE: Bottone "Mostra altri"
                    if let totalCount = msg.totalCount,
                       let currentLimit = msg.currentLimit,
                       currentLimit < totalCount,
                       currentLimit < 20 {
                        
                        let remaining = min(totalCount - currentLimit, 5)
                        
                        Button(action: {
                            vm.loadMore(for: msg.id)
                        }) {
                            HStack(spacing: 6) {
                                Image(systemName: "arrow.down.circle")
                                Text("Mostra altri \(remaining) vini")
                                    .font(.subheadline.weight(.medium))
                            }
                            .foregroundStyle(AppColors.primaryWine)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 12)
                            .background(
                                RoundedRectangle(cornerRadius: 12)
                                    .fill(AppColors.primaryWine.opacity(0.08))
                            )
                        }
                        .padding(.top, 8)
                    }
                }
                .padding(.top, 4)
                .transition(.opacity)
                .animation(.easeInOut(duration: 0.18), value: wines.count)
            }
        }
    }
// TRANCHE 2/2
    private func wineRow(_ wine: WineCard) -> some View {
        VStack(alignment: .leading, spacing: 10) {

            HStack(alignment: .top) {

                VStack(alignment: .leading, spacing: 6) {

                    HStack(spacing: 8) {
                        Text(wine.name)
                            .font(.headline)

                        if let rank = wine.rank {
                            Text("#\(rank)")
                                .font(.caption)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 2)
                                .background(rank == 1 ? Color.blue.opacity(0.18) : Color.gray.opacity(0.15))
                                .clipShape(Capsule())
                        }
                    }

                    // ✅ Overall bar (coerente col rank in sort=relevance) + Match label separata
                    let rawMatch = (wine.match_score ?? wine.__match_score) ?? 0.0
                    let matchClamped = max(0.0, min(1.0, rawMatch))
                    let overallClamped = max(0.0, min(1.0, (wine.score ?? 0) / 5.0))

                    GeometryReader { geo in
                        ZStack(alignment: .leading) {
                            Capsule()
                                .fill(Color.gray.opacity(0.18))
                                .frame(height: 6)

                            Capsule()
                                .fill(AppColors.accentWine)
                                .frame(width: geo.size.width * overallClamped, height: 6)
                        }
                    }
                    .frame(height: 6)
                    .padding(.top, 2)

                    HStack(spacing: 12) {

                        HStack(spacing: 6) {
                            Image(systemName: "chart.bar.fill")
                                .font(.caption.weight(.semibold))
                                .symbolRenderingMode(.hierarchical)
                                .foregroundStyle(AppColors.accentWine)

                            Text("Overall \(Int(overallClamped * 100))%")
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(.secondary)
                        }
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Color.gray.opacity(0.10))
                        .clipShape(Capsule())

                        // Match: mostrato solo se > 0, ma con stile più evidente
                        if rawMatch > 0.0001 {
                            HStack(spacing: 6) {
                                Image(systemName: "scope")
                                    .font(.caption.weight(.semibold))
                                    .symbolRenderingMode(.hierarchical)
                                    .foregroundStyle(.blue)

                                Text("Match \(Int(matchClamped * 100))%")
                                    .font(.caption.weight(.semibold))
                                    .foregroundStyle(.secondary)
                            }
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(Color.blue.opacity(0.10))
                            .clipShape(Capsule())
                        }
                    }

                    HStack(spacing: 10) {
                        if let p = wine.price {
                            Text(String(format: "€%.2f", p))
                                .font(.headline.weight(.semibold))
                                .foregroundStyle(AppColors.accentWine)
                        }

                        if let r = wine.rating_overall, r > 0 {
                            let clamped = max(0, min(5, r))
                            let full = Int(clamped.rounded(.down))
                            let hasHalf = (clamped - Double(full)) >= 0.5
                            let colorFilled = Color(red: 0.95, green: 0.82, blue: 0.35) // paglierino tenue

                            HStack(spacing: 3) {
                                ForEach(0..<5, id: \.self) { i in
                                    if i < full {
                                        Image(systemName: "star.fill")
                                            .foregroundStyle(colorFilled)
                                    } else if i == full && hasHalf {
                                        Image(systemName: "star.leadinghalf.filled")
                                            .foregroundStyle(colorFilled)
                                    } else {
                                        Image(systemName: "star")
                                            .foregroundStyle(Color.gray.opacity(0.35))
                                    }
                                }
                                Text(String(format: "%.1f", r))
                            }
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        }
                    }

                    // ✅ DEBUG: SEMPRE fuori dal prezzo/stelle (così non sballa layout e non dipende dal prezzo)
                    if rankingDebugMode {
                        HStack(spacing: 8) {
                            Text("match: \((wine.match_score ?? wine.__match_score ?? 0), specifier: "%.2f")")
                            Text("overall: \(((wine.score ?? 0)/5.0), specifier: "%.2f")")
                            Text("value: \((wine.__value_score ?? 0), specifier: "%.2f")")
                            Text("semantic: \((wine.__components?["__semantic_boost"] ?? 0), specifier: "%.2f")")
                        }
                        .font(.caption2)
                        .foregroundStyle(.gray)
                    }
                }

                Spacer()

                Button {
                    favoritesStore.toggle(wine)
                } label: {
                    Image(systemName: favoritesStore.isFavorite(wine) ? "heart.fill" : "heart")
                        .foregroundStyle(favoritesStore.isFavorite(wine) ? .red : .secondary)
                }
            }

            let metaTop = metaLineTop(wine)
            let metaBottom = metaLineBottom(wine)

            if !metaTop.isEmpty || !metaBottom.isEmpty {
                VStack(alignment: .leading, spacing: 2) {
                    if !metaTop.isEmpty {
                        Text(metaTop)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                    if !metaBottom.isEmpty {
                        Text(metaBottom)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                }
            }

            HStack(spacing: 6) {
                if let highlights = wine.ui_highlights, !highlights.isEmpty {
                    ForEach(Array(highlights.prefix(3)), id: \.self) { h in
                        badgeLight(h)
                    }
                } else {
                    if wine.reason.lowercased().contains("aperitivo") {
                        badge("Perfetto per aperitivo")
                    }

                    if wine.reason.lowercased().contains("cena") ||
                        wine.reason.lowercased().contains("importante") {
                        badge("Ideale cena importante")
                    }

                    if let score = wine.score, score > 4.4,
                       let price = wine.price, price < 35 {
                        badge("Top qualità/prezzo")
                    }
                }
            }
            
            // ✅ Explain Mode B: se presente, mostra solo la prima riga; altrimenti fallback su reason
            if let explain = wine.explain,
               let firstExplain = explain.first,
               !firstExplain.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                Text(firstExplain)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            } else {
                Text(wine.reason)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .lineLimit(3)
            }
            
            if wine.rank == 1 && ((wine.explain ?? []).isEmpty) {
                Text("Scelto come miglior match per la tua richiesta")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            if let pairings = wine.food_pairings, !pairings.isEmpty {
                Text("🍽 " + pairings.map { $0.replacingOccurrences(of: "_", with: " ").capitalized }.joined(separator: " • "))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }

            // ✅ UI-C: bottone espandi/collassa
            let isExpanded = expandedWineIDs.contains(wine.id)
            Button {
                if isExpanded { expandedWineIDs.remove(wine.id) }
                else { expandedWineIDs.insert(wine.id) }
            } label: {
                HStack {
                    Text(isExpanded ? "Meno dettagli" : "Più dettagli")
                        .font(.caption).foregroundStyle(AppColors.accentWine)
                    Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                        .font(.caption2).foregroundStyle(AppColors.accentWine)
                }
            }.buttonStyle(.plain)

            if isExpanded {
                let judge = judgementsInline(wine)
                if !judge.isEmpty {
                    Text(judge)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }

                if let url = wine.purchase_url, let u = URL(string: url) {
                    Link("Apri link acquisto", destination: u)
                        .font(.footnote)
                }

                if let tags = wine.tags, !tags.isEmpty {
                    Text(tags.filter { !["red","white","rose","rosso","bianco","rosato","ruby_red","ruby red","low","medium","high","fermo","secco","dolce","amabile","frizzante","spumante","straw yellow","straw_yellow","sparkling","sweet"].contains($0.lowercased()) }.map { $0.replacingOccurrences(of: "_", with: " ") }.joined(separator: " • "))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .padding(12)
        .background(AppColors.cardBackground)
        .clipShape(RoundedRectangle(cornerRadius: 14))
    }

    // MARK: - Formatting helpers

    private func metaLineTop(_ w: WineCard) -> String {
        var parts: [String] = []
        if let producer = clean(w.producer) { parts.append(producer) }
        let originParts = [clean(WineLocalizer.country(w.country)), clean(WineLocalizer.region(w.region)), clean(w.zone)].compactMap { $0 }
        if !originParts.isEmpty { parts.append(originParts.joined(separator: " · ")) }
        return parts.joined(separator: " · ")
    }

    private func metaLineBottom(_ w: WineCard) -> String {
        var parts: [String] = []
        if let denom = clean(w.denomination) { parts.append(denom.uppercased()) }
        if let v = w.vintage { parts.append(String(v)) }
        return parts.joined(separator: " · ")
    }

    private func judgementsInline(_ w: WineCard) -> String {
        var parts: [String] = []
        if let q = clean(w.quality) { parts.append("Qualità: \(q)") }
        if let b = clean(w.balance) { parts.append("Equilibrio: \(b)") }
        if let p = clean(w.persistence) { parts.append("Persistenza: \(p)") }
        if let c = clean(w.color_detail) { parts.append("Colore: \(WineLocalizer.color(c))") }
        return parts.joined(separator: " · ")
    }

    private func clean(_ s: String?) -> String? {
        guard let s = s?.trimmingCharacters(in: .whitespacesAndNewlines), !s.isEmpty else { return nil }
        return s
    }

    @ViewBuilder
    private func badge(_ text: String) -> some View {
        Text(text)
            .font(.caption2)
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(AppColors.accentWine.opacity(0.12))
            .clipShape(Capsule())
    }

    // ✅ Badge “light” per explainability (ui_highlights)
    @ViewBuilder
    private func badgeLight(_ text: String) -> some View {
        Text(text)
            .font(.caption2)
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(Color.gray.opacity(0.12))
            .clipShape(Capsule())
    }
    private func topReasonLabel(for wine: WineCard) -> String {

        let match = (wine.match_score ?? wine.__match_score) ?? 0.0
        let query = vm.messages.last(where: { $0.role == .user })?.text.lowercased() ?? ""

        // 🎯 Match perfetto
        if match >= 0.95 {
            return "🎯 Perfetto per la tua richiesta"
        }

        // 🍽 Occasione
        let fp = (wine.food_pairings ?? []).joined(separator: " ").replacingOccurrences(of: "_", with: " ").lowercased()
        let tags = (wine.tags ?? []).joined(separator: " ").lowercased()
        
        // 🍣 Sushi / pesce
        if query.contains("sushi") {
            if fp.contains("sushi") || fp.contains("pesce") || fp.contains("crudi") || fp.contains("crosta") {
                return "🍣 Perfetto con sushi e pesce"
            }
            // fallback soft: vini freschi/minerali spesso ottimi col sushi
            if tags.contains("fresco") || tags.contains("minerale") || tags.contains("sapido") {
                return "🍣 Ottimo con sushi (fresco e minerale)"
            }
        }
        
        // 🍽 Cena / occasione (inferita da tags + food_pairings)
        if query.contains("cena"),
           (tags.contains("cena") || tags.contains("importante") || fp.contains("carne") || fp.contains("selvaggina")) {
            return "🍽 Ideale per una cena importante"
        }

        // 🍷 Struttura
        if query.contains("struttur"),
           wine.intensity?.lowercased() == "high" {
            return "🍷 Profilo potente e strutturato"
        }

        // 💰 Prezzo/Valore
        if vm.selectedSort == .price_value,
           (wine.__value_score ?? 0) > 1.15 {
            return "💰 Eccellente qualità/prezzo"
        }

        return "⭐ Primo risultato per coerenza complessiva"
    }

    // MARK: - Grouping Annate (UI-only)

    private struct WineGroup: Identifiable {
        let id: String
        let representativeIndex: Int
        let representative: WineCard
        var vintages: [Int]
    }

    private struct WineGroupKey: Hashable {
        let denomination: String
        let zone: String
        let producer: String

        init?(groupableWine w: WineCard) {
            func norm(_ s: String?) -> String? {
                let t = s?.trimmingCharacters(in: .whitespacesAndNewlines)
                guard let t, !t.isEmpty else { return nil }
                return t.lowercased()
            }

            guard
                let d = norm(w.denomination),
                let z = norm(w.zone),
                let p = norm(w.producer)
            else { return nil }

            self.denomination = d
            self.zone = z
            self.producer = p
        }
    }

    private func groupedWinesForDisplay(_ wines: [WineCard]) -> [WineGroup] {
        var groupIndexByKey: [WineGroupKey: Int] = [:]
        var groups: [WineGroup] = []
        groups.reserveCapacity(wines.count)

        for (idx, w) in wines.enumerated() {
            guard let key = WineGroupKey(groupableWine: w) else {
                var g = WineGroup(
                    id: "ungrouped:\(w.id):\(idx)",
                    representativeIndex: idx,
                    representative: w,
                    vintages: []
                )
                if let v = w.vintage { g.vintages = [v] }
                groups.append(g)
                continue
            }

            if let gIndex = groupIndexByKey[key] {
                if let v = w.vintage, !groups[gIndex].vintages.contains(v) {
                    groups[gIndex].vintages.append(v)
                }
            } else {
                groupIndexByKey[key] = groups.count
                var g = WineGroup(
                    id: "\(key.denomination)|\(key.zone)|\(key.producer)",
                    representativeIndex: idx,
                    representative: w,
                    vintages: []
                )
                if let v = w.vintage { g.vintages = [v] }
                groups.append(g)
            }
        }

        return groups
    }

    private func vintagesLine(_ vintages: [Int]) -> String? {
        guard vintages.count > 1 else { return nil }

        let sorted = vintages.sorted()
        var unique: [Int] = []
        unique.reserveCapacity(sorted.count)

        var last: Int? = nil
        for v in sorted {
            if last != v {
                unique.append(v)
                last = v
            }
        }

        guard unique.count > 1 else { return nil }
        let joined = unique.map(String.init).joined(separator: " • ")
        return "Annate disponibili: \(joined)"
    }
}

// MARK: - Preferences

private struct BottomYKey: PreferenceKey {
    static var defaultValue: CGFloat = .greatestFiniteMagnitude
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) { value = nextValue() }
}

private struct ViewportHeightKey: PreferenceKey {
    static var defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) { value = nextValue() }
}

private struct ScrollOffsetYKey: PreferenceKey {
    static var defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) { value = nextValue() }
}

// MARK: - Seeded shuffle

private struct SeededGenerator: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed == 0 ? 0xdeadbeef : seed
    }

    mutating func next() -> UInt64 {
        state = 6364136223846793005 &* state &+ 1
        return state
    }
}

// MARK: - Grape Picker Sheet (UI-only)

private struct GrapePickerSheet: View {

    let title: String
    let options: [String]
    @Binding var selection: String

    @Environment(\.dismiss) private var dismiss
    @State private var searchText: String = ""

    private var safeOptions: [String] {
        options.isEmpty ? ["Tutti"] : options
    }

    private var normalizedOptions: [String] {
        var seen = Set<String>()
        return safeOptions.filter { opt in
            let k = opt.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            guard !k.isEmpty else { return false }
            if seen.contains(k) { return false }
            seen.insert(k)
            return true
        }
    }

    private var filtered: [String] {
        let q = searchText.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !q.isEmpty else { return normalizedOptions }
        return normalizedOptions.filter { $0.lowercased().contains(q) }
    }

    private var sections: [(String, [String])] {
        var list = filtered
        let tutti = list.first(where: { $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() == "tutti" })
        list.removeAll { $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() == "tutti" }

        var map: [String: [String]] = [:]
        for g in list {
            let s = g.trimmingCharacters(in: .whitespacesAndNewlines)
            let letter = s.first.map { String($0).uppercased() } ?? "#"
            map[letter, default: []].append(g)
        }

        let keys = map.keys.sorted { $0.localizedCaseInsensitiveCompare($1) == .orderedAscending }
        var out: [(String, [String])] = []
        if let tutti { out.append(("", [tutti])) }

        for k in keys {
            let values = (map[k] ?? []).sorted { $0.localizedCaseInsensitiveCompare($1) == .orderedAscending }
            out.append((k, values))
        }
        return out
    }

    var body: some View {
        NavigationStack {
            List {
                if !searchText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    ForEach(filtered, id: \.self) { g in
                        row(g)
                    }
                } else {
                    ForEach(sections, id: \.0) { sec in
                        if sec.0.isEmpty {
                            Section {
                                ForEach(sec.1, id: \.self) { g in row(g) }
                            }
                        } else {
                            Section(sec.0) {
                                ForEach(sec.1, id: \.self) { g in row(g) }
                            }
                        }
                    }
                }
            }
            .searchable(
                text: $searchText,
                placement: .navigationBarDrawer(displayMode: .always),
                prompt: "Cerca vitigno"
            )
            .navigationTitle(title)
            .navigationBarTitleDisplayMode(.inline)
        }
    }

    @ViewBuilder
    private func row(_ grape: String) -> some View {
        Button {
            selection = grape
            dismiss()
        } label: {
            HStack {
                Text(grape)

                Spacer()

                if grape.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
                    == selection.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {

                    Image(systemName: "checkmark")
                        .foregroundStyle(.secondary)
                }
            }
        }
    }
}

// MARK: - Simple Picker Sheet (UI-only)

private struct SimplePickerSheet: View {
    let title: String
    let options: [String]
    @Binding var selection: String

    @Environment(\.dismiss) private var dismiss

    private var safeOptions: [String] {
        options.isEmpty ? ["Tutti"] : options
    }

    var body: some View {
        NavigationStack {
            List {
                ForEach(safeOptions, id: \.self) { opt in
                    Button {
                        selection = opt
                        dismiss()
                    } label: {
                        HStack {
                            Text(opt)

                            Spacer()

                            if opt.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
                                == selection.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {

                                Image(systemName: "checkmark")
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                }
            }
            .navigationTitle(title)
            .navigationBarTitleDisplayMode(.inline)
        }
    }
}
