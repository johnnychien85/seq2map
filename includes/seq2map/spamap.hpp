#ifndef SPAMAP_HPP
#define SPAMAP_HPP

#include <boost/enable_shared_from_this.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <seq2map/common.hpp>

namespace seq2map
{
namespace spamap
{
    template<typename T>
    class Node : public boost::enable_shared_from_this<Node<T> >
    {
    public:
        class Row;
        class Column;
        typedef Column Col;

        virtual ~Node()
        {
            // Make sure the Node has been fully detached when it's being destroyed.
            // Node deletion should only be invoked automatically by smart pointers.
            assert(up.expired());
            assert(down.expired());
            assert(left.expired());
            assert(!right);

            //E_TRACE << ToString() << " destroyed";
        }

        inline String ToString() const
        {
            std::stringstream ss;
            ss << "(" << row.GetIndex() << "," << col.GetIndex() << ")";

            return ss.str();
        }

    private:
        typedef Node<T> NodeType;
        typedef boost::shared_ptr<NodeType> Own;
        typedef boost::weak_ptr  <NodeType> Ref;

        // can only be constructed by Row::Insert
        Node(Row& row, Col& col, const T& value) : row(row), col(col), value(value), indexed(false)
        {
            //E_TRACE << ToString() << " born";
        }

    public:
        Row& row;
        Col& col;
        T value;

        Ref up;
        Ref down;
        Ref left;
        Own right;

        bool indexed;

    public:
        class Header : public Indexed
        {
        public:
            Header(size_t idx) : Indexed(idx) {}
            virtual bool empty() const { return !m_head; }
            virtual void clear() = 0;
        
        protected:
            Own m_head;
            Own m_tail;

        public: // STL-compliance members
            typedef T value_type;
        };

        class Iterator
        : public boost::iterator_facade<Iterator, NodeType, boost::bidirectional_traversal_tag, T&>
        {
        public:
            Iterator() {}
            explicit Iterator(Own& node) : m_node(node) {}

            bool operator==(Iterator const& itr) const
            {
                return equal(itr);
            }

        protected:
            friend class boost::iterator_core_access;

            bool equal(Iterator const& itr) const
            {
                return m_node == itr.m_node;
            }

            T& dereference() const { return m_node->value; }

            virtual void increment() = 0;
            virtual void decreasement() = 0;

            Own m_node;
        };

        class Row : public Header
        {
        public:
            Row(size_t idx = INVALID_IDX) : Header(idx) {}
            ~Row() { clear(); }

            T& Insert(Col& col, const T& value)
            {
                Own node = Own(new NodeType(*this, col, value));
                Own prev = m_tail;
                Own next;

                if (empty())
                {
                    assert(!m_head);
                    m_head = m_tail = node->shared_from_this();
                }
                else
                {
                    while (prev && node->col < prev->col)
                    {
                        next = prev;
                        prev = prev->left.lock();
                    }

                    if (next) // take over the ownership
                    {
                        node->right = next;
                        next->left  = node->shared_from_this();

                        // E_TRACE << node->ToString() << " <----> " << next->ToString();
                    }
                    else
                    {
                        m_tail = node->shared_from_this();
                    }

                    if (prev)
                    {
                        node->left  = prev;
                        prev->right = node->shared_from_this();

                        // E_TRACE << prev->ToString() << " <----> " << node->ToString();
                    }
                    else
                    {
                        m_head = node->shared_from_this();
                    }
                }

                col.Insert(node, prev);

                return node->value;
            }

            void Remove(Own node)
            {
                assert(node);
                assert(&node->row == this); // ownership check

                node->col.Remove(node);

                Own prev = node->left.lock();
                Own next = node->right;

                if (prev)
                {
                    prev->right = next;
                    // E_TRACE << prev->ToString() << " <-//-> " << node->ToString();
                }
                else
                {
                    assert(m_head == node);
                    m_head = next;
                }

                if (next)
                {
                    next->left = prev;
                    // E_TRACE << node->ToString() << " <-//-> " << next->ToString();
                }
                else
                {
                    assert(m_tail == node);
                    m_tail = prev;
                }

                // detach the node completely
                node->up = node->down = node->left = node->right = Own();
            }

        public: // STL-compliance members
            class Iterator : public NodeType::Iterator
            {
            public:
                Iterator(Own& node = Own()) : NodeType::Iterator(node) {}
            protected:
                virtual void increment() { m_node = m_node->right; }
                virtual void decreasement() { m_node = m_node->left.lock(); }
            };

            typedef Iterator iterator;

            inline iterator begin() { return Iterator(m_head); }
            inline iterator end()   { return Iterator(); }

            virtual inline void clear()
            {
                while (m_tail) Remove(m_tail);
                assert(empty());
            }
        };

        class Column : public Header
        {
        public:
            Column(size_t idx = INVALID_INDEX) : Header(idx) {}
            ~Column() { clear(); }

        protected:
            friend class Row;

            void Insert(Own& node, Own& hint)
            {
                assert(&node->col == this); // ownership check

                const size_t row = node->row.GetIndex();
                const size_t col = node->col.GetIndex();

                Own prev;
                Own next;

                // check if the hint can be used or not
                bool hit = hint && hint->col.GetIndex() == col;

                if (hit)
                {
                    // structural checks
                    assert(hint->row.GetIndex() == row);
                    assert(!empty());

                    prev = hint;
                    next = hint->down.lock(); // could be null
                }
                else if(!empty()) // find the predecessor and successor if there's any
                {
                    // find the first element whose row is later than the new node
                    NodeRefs::iterator itr = m_nodes.lower_bound(row);

                    if (itr != m_nodes.end()) // got a successor
                    {
                        next = itr->second.lock(); // the successor
                        prev = next ? next->up.lock() : Own(); // the predeccessor, could be null

                        // neither successor nor predeccessor (if exists) can be at the sam
                        // or it should have been detected and passed as a hint by Row::Insert()
                        assert( next && next->row.GetIndex() != row);
                        assert(!prev || prev->row.GetIndex() != row);
                    }
                    else // insert to the end of column
                    {
                        itr--; // move to the last element

                        prev = itr->second.lock(); // the predeccessor
                        next = prev ? prev->down.lock() : Own(); // the successor, has to be null

                        assert( prev && prev->row.GetIndex() != row); // see the other case
                        assert(!next); // the end of column has nothing down there
                    }
                }

                if (prev)
                {
                    prev->down = node->shared_from_this();
                    node->up   = prev;

                    //E_TRACE << prev->ToString() << " <----> " << node->ToString();
                }
                else
                {
                    m_head = node->shared_from_this();
                }

                if (next)
                {
                    next->up   = node->shared_from_this();
                    node->down = next;

                    //E_TRACE << node->ToString() << " <----> " << next->ToString();
                }
                else
                {
                    m_tail = node->shared_from_this();
                }

                if (!hit) // update the index for the first hit
                {
                    m_nodes[row] = node->shared_from_this();
                    node->indexed = true;
                }
            }

            void Remove(Own& node)
            {
                assert(node);
                assert(&node->col == this);

                const size_t row = node->row.GetIndex();

                Own prev = node->up.lock();
                Own next = node->down.lock();

                if (node->indexed)
                {
                    Own newIndex;

                    if      (prev && prev->row.GetIndex() == row) newIndex = prev;
                    else if (next && next->row.GetIndex() == row) newIndex = next;

                    if (newIndex)
                    {
                        newIndex->indexed = true;
                        m_nodes[row] = newIndex;
                    }
                    else
                    {
                        size_t erased = m_nodes.erase(row);
                        assert(erased == 1);
                    }

                    node->indexed = false;
                }

                if (prev)
                {
                    prev->down = next;
                    // E_TRACE << prev->ToString() << " <-//-> " << node->ToString();
                }
                else
                {
                    m_head = next;
                }

                if (next)
                {
                    next->up = prev;
                    // E_TRACE << node->ToString() << " <-//-> " << next->ToString();
                }
                else
                {
                    m_tail = prev;
                }
            }

        private:
            typedef std::map<size_t, Ref> NodeRefs;
            NodeRefs m_nodes;

        public: // STL-compliance members
            class Iterator : public NodeType::Iterator
            {
            public:
                Iterator(Own& node = Own()) : NodeType::Iterator(node) {}
            protected:
                virtual void increment() { m_node = m_node->down.lock(); }
                virtual void decreasement() { m_node = m_node->up.lock(); }
            };

            typedef Iterator iterator;

            inline iterator begin() { return Iterator(m_head); }
            inline iterator end()   { return Iterator(); }

            virtual inline void clear()
            {
                while (m_tail) m_tail->row.Remove(m_tail);

                // the indexing tree should be empty now
                assert(m_nodes.empty());
            }
        };
    };

    template<typename T, class R = Node<T>::Row, class C = Node<T>::Col>
    class Map
    {
    public:
        typedef R RowType;
        typedef C ColType;

        Map() {}

        virtual ~Map()
        {
            m_cols.clear(); // remove columns first
            m_rows.clear(); // then rows..
        }

        R& Row(size_t i)
        {
            Rows::iterator itr = m_rows.find(i);
            return itr != m_rows.end() ? itr->second :
                m_rows.insert(Rows::value_type(i, R(i))).first->second;
        }

        C& Col(size_t j)
        {
            Cols::iterator itr = m_cols.find(j);
            return itr != m_cols.end() ? itr->second :
                m_cols.insert(Cols::value_type(j, C(j))).first->second;
        }

        inline T& Insert(size_t i, size_t j, const T& value)
        {
            return Row(i).Insert(Col(j), value);
        }

        //inline T& operator(size_t i, size_t j)
        //{
        //    return Insert(i, j, T());
        //}

    private:
        typedef std::map<size_t, R> Rows;
        typedef std::map<size_t, C> Cols;

        Rows m_rows;
        Cols m_cols;
    };

    template<typename T, size_t dims>
    class NodeND
    {
    private:
        typedef NodeND<T, dims> NodeType;
        typedef NodeType* NodePtr;
        typedef const NodeType* ConstNodePtr;

    public:
        /**
         * The container is for internal use only to hold linked nodes.
         * The idea of container will be extended by dimensions.
         */
        class Container : public Indexed
        {
        public:
            Container(size_t idx) : Indexed(idx), m_head(NULL), m_tail(NULL) {}
            virtual String ToString() const = 0;

        public: // STL-compliance members
            virtual void insert(NodePtr node) = 0;
            virtual void  erase(NodePtr node) = 0;
            virtual bool  empty() const { return m_head == NULL; }
            virtual void  clear() { while (!empty()) erase(m_tail); }

        protected:
            NodePtr m_head;
            NodePtr m_tail;
        };

    private:

        /**
         * Link in a specific dimension
         */
        struct Link
        {
            Link() : d(NULL), prev(NULL), next(NULL), indexed(false) {}

            Container* d;
            NodePtr prev;
            NodePtr next;
            bool indexed;
        };

        NodeND(const T& value) : m_value(value) {}

        virtual ~NodeND()
        {
            // E_TRACE << "destroying " << ToString();
        }

        String ToString() const
        {
            std::stringstream ss;

            ss << "(";
            for (size_t d = 0; d < dims; d++)
            {
                ss << m_links[d].d->GetIndex() << (d < dims - 1 ? "," : "");
            }
            ss << ")";

            return ss.str();
        }

        /*
        bool operator< (const NodeType& rhs) const
        {
            const NodeType& lhs = *this;
            for (size_t d = 0; d < dims; d++)
            {
                if (*(lhs.m_links[d].d) < *(rhs.m_links[d].d))
                {
                    return true;
                }
            }
            return false;
        }

        bool operator== (const NodeType& node) const
        {
            const NodeType& lhs = *this;
            for (size_t d = 0; d < dims; d++)
            {
                if (*(lhs.m_links[d].d) != *(rhs.m_links[d].d))
                {
                    return false;
                }
            }
            return true;
        }
        */

        Link m_links[dims];
        T    m_value;

    public:
        /**
         * Dimension is the externally extensible class.
         */
        template<size_t dim, size_t idepth = dims - 1>
        class Dimension : public Container
        {
        public:
            Dimension(size_t idx) : Container(idx) {}

            String ToString() const
            {
                std::stringstream ss;

                ss << "D" << dim << "[" << GetIndex() << "] : ";
                ss << "head -> ";
                for (const_iterator itr = cbegin(); itr; itr++)
                {
                    ss << itr.m_node->ToString() << (itr.m_node->m_links[dim].indexed ? "*" : "") << " -> ";
                }
                ss << "tail";

                return ss.str();
            }

        private: // some data types needed for indexing and search
            /**
             * Node used in indexed search phase.
             */
            struct INode
            {
                NodePtr node;
                INode(const NodePtr node = NULL) : node(node) {}

                bool operator< (const INode& rhs) const
                {
                    static const size_t d0 = dim < idepth ? (idepth + 1) : idepth;
                    const INode& lhs = *this;

                    assert(d0 <= dims);

                    for (size_t d = 0; d < d0; d++)
                    {
                        if (*(lhs.node->m_links[d].d) < *(rhs.node->m_links[d].d))
                        {
                            //E_TRACE << lhs.node->ToString() << " < " << rhs.node->ToString() << ", d = " << d;
                            return true;
                        }
                        else if (*(rhs.node->m_links[d].d) < *(lhs.node->m_links[d].d))
                        {
                            //E_TRACE << rhs.node->ToString() << " < " << lhs.node->ToString() << ", d = " << d;
                            return false;
                        }
                    }
                    //E_TRACE << lhs.node->ToString() << " not less than " << rhs.node->ToString() << ", d0 = " << d0;
                    return false;
                }

                bool operator== (const INode& rhs) const
                {
                    static const size_t d0 = dim < idepth ? (idepth + 1) : idepth;
                    const INode& lhs = *this;

                    assert(d0 <= dims);

                    for (size_t d = 0; d < d0; d++)
                    {
                        if (!(*(lhs.node->m_links[d].d) == *(rhs.node->m_links[d].d)))
                        {
                            return false;
                        }
                    }
                    return true;
                }
            };

            // node used in linear search phase
            struct LNode
            {
                NodePtr node;
                LNode(const NodePtr node = NULL) : node(node) {}

                bool operator< (const LNode& rhs) const
                {
                    static const size_t d0 = dim < idepth ? (idepth + 1) : idepth;
                    const LNode& lhs = *this;

                    for (size_t d = d0; d < dims; d++)
                    {
                        if ((*lhs.node->m_links[d].d) < (*rhs.node->m_links[d].d))
                        {
                            //E_TRACE << lhs.node->ToString() << " < " << rhs.node->ToString() << ", d = " << d;
                            return true;
                        }
                        else if ((*rhs.node->m_links[d].d) < (*lhs.node->m_links[d].d))
                        {
                            //E_TRACE << rhs.node->ToString() << " < " << lhs.node->ToString() << ", d = " << d;
                            return false;
                        }
                    }
                    //E_TRACE << lhs.node->ToString() << " not less than " << rhs.node->ToString() << ", d0 = " << d0;
                    return false;
                }
            };

            typedef std::map<INode, NodePtr> Indexer;
            Indexer m_imap;

        protected: // STL-compliance members
            virtual void insert(NodePtr node)
            {
                Link& link = node->m_links[dim];

                assert(link.d == this); // ownership assertion

                NodePtr prev = upper_bound(node);
                NodePtr next = prev == NULL ? m_head : prev->m_links[dim].next;

                if (prev == NULL)
                {
                    m_head = node;
                }
                else
                {
                    prev->m_links[dim].next = node;
                }

                if (next == NULL)
                {
                    m_tail = node;
                }
                else
                {
                    next->m_links[dim].prev = node;
                }

                link.prev = prev;
                link.next = next;

                NodePtr& indexed = m_imap[INode(node)];

                if (indexed == NULL || !(LNode(node) < LNode(indexed)))
                {
                    if (indexed != NULL)
                    {
                        indexed->m_links[dim].indexed = false;
                    }

                    indexed = node;
                    node->m_links[dim].indexed = true;

                    // E_TRACE << "indexed " << node->ToString();
                }

                /*
                if (prev != NULL && !(INode(prev) < INode(node)))
                {
                    assert(prev->m_links[dim].indexed);

                    prev->m_links[dim].indexed = false;
                    node->m_links[dim].indexed = true;

                    m_imap[INode(node)] = node;
                }*/
            }

            virtual void erase(NodePtr node)
            {
                Link& link = node->m_links[dim];

                if (link.prev != NULL) link.prev->m_links[dim].next = link.next;
                if (link.next != NULL) link.next->m_links[dim].prev = link.prev;

                if (node == m_head) m_head = link.next;
                if (node == m_tail) m_tail = link.prev;

                if (link.indexed)
                {
                    INode inode(node);

                    if (link.prev != NULL && (INode(link.prev) == inode))
                    {
                        assert(!link.prev->m_links[dim].indexed);
                        link.prev->m_links[dim].indexed = true;

                        //E_TRACE << "indexed " << link.prev->ToString();

                        m_imap[inode] = link.prev;
                    }
                    else
                    {
                        m_imap.erase(inode);
                    }
                }
            }

            NodePtr upper_bound(NodePtr node) const
            {
                if (empty()) return NULL;

                //E_TRACE << node->ToString();

                INode inode(node);
                LNode lnode(node);

                // try last element first - O(1) complexity
                if (INode(m_tail) < inode)
                {
                    //E_TRACE << "insert after tail " << m_tail->ToString();
                    return m_tail;
                }

                // try first element - O(1) complexity
                if (inode < INode(m_head))
                {
                    //E_TRACE << "insert before head " << m_head->ToString();
                    return NULL;
                }

                // perform indexed linear search - O(lgN+M) complexity in worst case
                Indexer::const_iterator itr = m_imap.lower_bound(inode);

                NodePtr rb = (itr == m_imap.cend())   ? m_tail : itr->second;
                NodePtr lb = (itr == m_imap.cbegin()) ? m_head : (--itr)->second;

                //E_TRACE << "bound : " << lb->ToString() << " - " << rb->ToString();

                if (inode < INode(rb))
                {
                    //E_TRACE << "insert after left bound";
                    return lb;
                }

                // do in-dimension backward linear search
                for (NodePtr prev = rb; prev != lb; prev = prev->m_links[dim].prev)
                {
                    //E_TRACE << "comparing " << prev->ToString();
                    if (LNode(prev) < lnode) return prev;
                }

                //E_TRACE << "insert after left bound after linear search";
                return lb;
            }

        public: // STL-compliance members
            typedef T value_type;

            template<typename node_ptr, typename value_type>
            class Iterator
            : public boost::iterator_facade<Iterator<node_ptr, value_type>, NodeType, boost::bidirectional_traversal_tag, T&>
            {
            public:
                Iterator() {}
                explicit Iterator(node_ptr node) : m_node(node) {}
                bool operator==(Iterator const& itr) const { return equal(itr); }

                inline operator bool() const { return m_node != NULL; }

            protected:
                friend class boost::iterator_core_access;
                friend class Dimension;

                bool equal(Iterator const& itr) const { return m_node == itr.m_node; }
                value_type& dereference() const { return m_node->value; }
                virtual void increment()    { m_node = (m_node == NULL) ? NULL : m_node->m_links[dim].next; }
                virtual void decreasement() { m_node = (m_node == NULL) ? NULL : m_node->m_links[dim].prev; }

                node_ptr m_node;
            }; // end of Iterator

            typedef Iterator<NodePtr, T> iterator;
            typedef Iterator<ConstNodePtr, T const> const_iterator;

            iterator begin() { return iterator(m_head); }
            iterator end()   { return iterator(m_tail); }

            const_iterator cbegin() const { return const_iterator(m_head); }
            const_iterator cend()   const { return const_iterator(m_tail); }

        }; // end of Dimension

        template<size_t idepth>
        class DimensionZero : public Dimension<0, idepth>
        {
        public:
            DimensionZero(size_t idx) : Dimension(idx) {}

            virtual ~DimensionZero()
            {
                while (!empty())
                {
                    NodePtr node = m_tail;
                    Remove(node);
                }
            }

            T& Insert(const T& value, Container** dn)
            {
                NodePtr node = new NodeType(value);

                for (size_t d = 0; d < dims; d++)
                {
                    node->m_links[d].d = (d == 0) ? static_cast<Container*>(this) : dn[d - 1];
                }

                //E_TRACE << "inserting " << node->ToString();

                // link the node dimension-by-dimension
                for (size_t d = 0; d < dims; d++)
                {
                    //E_TRACE << node->m_links[d].d->ToString();
                    node->m_links[d].d->insert(node);
                    //E_TRACE << node->m_links[d].d->ToString();
                }

                return node->m_value;
            }

        protected:
            void Remove(NodePtr node)
            {
                //E_TRACE << "removing " << node->ToString();

                // unlink the node dimension-by-dimension
                for (size_t d = 0; d < dims; d++)
                {
                    //E_TRACE << node->m_links[d].d->ToString();
                    node->m_links[d].d->erase(node);
                    //E_TRACE << node->m_links[d].d->ToString();
                }

                delete node;
            }
        };
    };

    template<
        typename T,
        class D0 = NodeND<T,3>::DimensionZero<1>,
        class D1 = NodeND<T,3>::Dimension<1,1>,
        class D2 = NodeND<T,3>::Dimension<2,1>
    >
    class Map3
    {
    public:
        Map3() {}

        virtual ~Map3()
        {
            m_dim0.clear();
        }

        D0& Dim0(size_t i) { return Dim<D0, S0>(m_dim0, i); }
        D1& Dim1(size_t j) { return Dim<D1, S1>(m_dim1, j); }
        D2& Dim2(size_t k) { return Dim<D2, S2>(m_dim2, k); }

        inline T& Insert(size_t i, size_t j, size_t k, const T& value)
        {
            typedef NodeND<T, 3>::Container* dnType;
            dnType d12[2];
            d12[0] = static_cast<dnType>(&Dim1(j));
            d12[1] = static_cast<dnType>(&Dim2(k));

            return Dim0(i).Insert(value, d12);
        }

    private:
        typedef std::map<size_t, D0> S0;
        typedef std::map<size_t, D1> S1;
        typedef std::map<size_t, D2> S2;

        template<typename D, typename S>
        D& Dim(S& dims, size_t idx)
        {
            S::iterator itr = dims.find(idx);
            return itr != dims.end() ? itr->second :
                dims.insert(S::value_type(idx, D(idx))).first->second;
        }

        S0 m_dim0;
        S1 m_dim1;
        S2 m_dim2;
    };
}
}
#endif // SPAMAP_HPP
