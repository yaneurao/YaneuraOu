COMPILER = g++
#COMPILER = mpicxx
CFLAGS   = -std=c++14 -fno-exceptions -fno-rtti -Wextra -Ofast -MMD -MP -fopenmp -fpermissive -D_WIN64
CFLAGS   += -march=native
LDFLAGS  = -lpthread #-lboost_mpi -lboost_serialization
LIBS     =
INCLUDE  = #-I../include
ifeq ($(OS),Windows_NT)
  TARGET = YaneuraOu-2016-Mid_gcc.exe
  LDFLAGS += -static
else
  TARGET = yaneuraou
endif
OBJDIR   = ../obj
ifeq "$(strip $(OBJDIR))" ""
  OBJDIR = ..
endif
#SOURCES  = $(wildcard *.cpp)
SOURCES  = bitboard.cpp misc.cpp movegen.cpp position.cpp shogi.cpp \
	thread.cpp tt.cpp usi.cpp \
	extra\bitop.cpp \
	extra\book.cpp \
	extra\entering_king_win.cpp \
	extra\long_effect.cpp \
	extra\mate1ply_with_effect.cpp \
	extra\mate1ply_without_effect.cpp \
	extra\move_picker_2016Q2.cpp \
	extra\test_cmd.cpp \
	extra\timeman.cpp \
	extra\see.cpp \
	eval\evaluate_bona_piece.cpp \
	eval\evaluate_kpp.cpp \
	eval\evaluate_kppt.cpp \
	eval\evaluate_material.cpp \
	eval\evaluate_no_eval.cpp \
	engine\2016-mid-engine\2016-mid-search.cpp \
	engine\user-engine\user-engine.cpp

OBJECTS  = $(addprefix $(OBJDIR)/, $(SOURCES:.cpp=.o))
DEPENDS  = $(OBJECTS:.o=.d)

$(TARGET): $(OBJECTS) $(LIBS)
	$(COMPILER) -o $@ $^ $(LDFLAGS) $(CFLAGS)

$(OBJDIR)/%.o: %.cpp
	@[ -d $(OBJDIR) ] || mkdir -p $(OBJDIR)
	$(COMPILER) $(CFLAGS) $(INCLUDE) -o $@ -c $<

all: clean $(TARGET)

avx2:
	mkdir -p ../obj/extra
	mkdir -p ../obj/eval
	mkdir -p ../obj/engine/2016-mid-engine/
	mkdir -p ../obj/engine/user-engine/
	$(MAKE) CFLAGS='$(CFLAGS) -DNDEBUG -mbmi2 -mavx2' LDFLAGS='$(LDFLAGS) -flto' $(TARGET)

sse4:
	mkdir -p ../obj/extra
	mkdir -p ../obj/eval
	mkdir -p ../obj/engine/2016-mid-engine/
	mkdir -p ../obj/engine/user-engine/
	$(MAKE) CFLAGS='$(CFLAGS) -DNDEBUG -msse4.2' LDFLAGS='$(LDFLAGS) -flto' $(TARGET)

sse2:
	mkdir -p ../obj/extra
	mkdir -p ../obj/eval
	mkdir -p ../obj/engine/2016-mid-engine/
	mkdir -p ../obj/engine/user-engine/
	$(MAKE) CFLAGS='$(CFLAGS) -DNDEBUG -msse2' LDFLAGS='$(LDFLAGS) -flto' $(TARGET)

prof:
	$(MAKE) CFLAGS='$(CFLAGS) -pg' bmi2

profgen:
	$(MAKE) CFLAGS='$(CFLAGS) -fprofile-generate -lgcov' LDFLAGS='$(LDFLAGS) -fprofile-generate -lgcov' bmi2

profuse:
	$(MAKE) CFLAGS='$(CFLAGS) -fprofile-use -lgcov' LDFLAGS='$(LDFLAGS) -fprofile-use -lgcov' bmi2

pgo:
	$(MAKE) profgen
	@./$(TARGET) bench
	@touch $(SOURCES)
	$(MAKE) profuse

clean:
	rm -f $(OBJECTS) $(DEPENDS) $(TARGET) ${OBJECTS:.o=.gcda}

-include $(DEPENDS)
